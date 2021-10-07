function [detected_sequence_vecs] = runViterbiDetection_controller_unaware(viterbiParams,SMdata)
parforProgress = parallel.pool.DataQueue;
afterEach(parforProgress, @updateParforProgress);
percentDone = nan;
reverseStr = nan;
timeRemaining = nan;
proc_start = nan;
last_time_updated = nan;
msg1 = '';
print_progress = true;

numMCevalHorizons = viterbiParams.numMCevalHorizons;
h_vec_num = viterbiParams.h_vec_num;
k_num = viterbiParams.k_num;
horizonsPerDay = viterbiParams.horizonsPerDay;
k_num_idxs_in_horizons = viterbiParams.k_num_idxs_in_horizons;
minPowerDemandInW = viterbiParams.minPowerDemandInW;
P_Hk_vec_gHkn1_vec = viterbiParams.P_Hk_vec_gHkn1_vec;
minLogParam = viterbiParams.minLogParam;
x_num = viterbiParams.x_num;
p_pu = viterbiParams.p_pu;
x_offset = viterbiParams.x_offset;
P_XkgXkn1HkHkn1 = viterbiParams.P_XkgXkn1HkHkn1;
P_H1_vec_X1 = viterbiParams.P_H1_vec_X1;

lP_XkgXkn1HkHkn1 = max(log(P_XkgXkn1HkHkn1),minLogParam);
lP_Hk_vec_gHkn1_vec = max(log(P_Hk_vec_gHkn1_vec),minLogParam);

lP_H1_vec_X1 = max(log(P_H1_vec_X1),minLogParam);
 
x_k_idxs = min(max(1,round((SMdata-minPowerDemandInW)/p_pu)-x_offset),x_num);
detected_sequence_vecs = zeros(numMCevalHorizons,k_num);

fprintf('\t\t\tRunning Viterbi MAP detection without controller : ');
initializeParforProgress;
incPercent = (1/numMCevalHorizons)*100;
parfor day_idx = 1:numMCevalHorizons
    k_num_idxs_in_horizons_t = k_num_idxs_in_horizons;
    lP_Hk_vec_gHkn1_vec_t = lP_Hk_vec_gHkn1_vec;
    lP_XkgXkn1HkHkn1_t = lP_XkgXkn1HkHkn1;
    x_k_idxs_t = x_k_idxs(day_idx,:);
    lP_H1_vec_X1_t = lP_H1_vec_X1;
        
    detected_sequence_t = zeros(1,k_num);
    pointers = zeros(h_vec_num,k_num);
    for horizonIdx = horizonsPerDay:-1:1
        k_idx_t = k_num_idxs_in_horizons_t(1,horizonIdx);
        cur_x_idx = x_k_idxs_t(k_idx_t);        
        max_prob_so_far =  lP_H1_vec_X1_t(:,cur_x_idx,horizonIdx);
        prev_x_idx = cur_x_idx;
        for k_idx = k_num_idxs_in_horizons_t(2:end,horizonIdx)'
            cur_x_idx = x_k_idxs_t(k_idx);
            max_prob_so_far_t = zeros(h_vec_num,1);
            for h_vec_idx = 1:h_vec_num
                temp = max_prob_so_far + lP_Hk_vec_gHkn1_vec_t(h_vec_idx,:,horizonIdx)' + reshape(lP_XkgXkn1HkHkn1_t(cur_x_idx,prev_x_idx,h_vec_idx,:),[],1);
                [max_prob_so_far_t(h_vec_idx), pointers(h_vec_idx,k_idx)] = max(temp);
            end
            max_prob_so_far =  max_prob_so_far_t;
            prev_x_idx = cur_x_idx;
        end
        
        k_idx_t = k_num_idxs_in_horizons_t(end,horizonIdx);
        [~,detected_sequence_t(k_idx_t)] = max(max_prob_so_far);
        for k_idx=k_num_idxs_in_horizons_t(end,horizonIdx)-1:-1:k_num_idxs_in_horizons_t(1,horizonIdx)
            detected_sequence_t(k_idx) = pointers(detected_sequence_t(k_idx+1),k_idx+1);            
        end
    end
    
    detected_sequence_vecs(day_idx,:) = detected_sequence_t';
    send(parforProgress, incPercent);
end
terminateParforProgress;

    function initializeParforProgress(print_progress_t)
        if(nargin==1)
            print_progress = print_progress_t;
        end
        percentDone = 0;
        timeRemaining = nan;
        proc_start = tic;
        last_time_updated = -inf;
        if(print_progress)
            reverseStr = 'Percent done = ';
            msg = sprintf('%3.2f', percentDone);
            msg1 = sprintf(', Est. time left = %d',timeRemaining);
            msg = [msg,msg1];
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
        end
    end

    function updateParforProgress(incPercent)
        percentDone = percentDone + incPercent;
        proc_time = toc(proc_start);
        if(proc_time-last_time_updated >1)
            if(print_progress)
                msg = sprintf('%3.2f', percentDone);
                timeRemaining = round((proc_time*(100-percentDone)/percentDone));
                if(timeRemaining>86400)
                    msg1 = sprintf('; Est. time left = %3.1f days   ',timeRemaining/86400);
                elseif(timeRemaining>3600)
                    msg1 = sprintf('; Est. time left = %3.1f hours ',timeRemaining/3600);
                elseif(timeRemaining>60)
                    msg1 = sprintf('; Est. time left = %d minutes',round(timeRemaining/60));
                else
                    msg1 = sprintf('; Est. time left = %d seconds',timeRemaining);
                end
                msg = [msg,msg1];
                fprintf([reverseStr, msg]);
                reverseStr = repmat(sprintf('\b'), 1, length(msg));
            end
            last_time_updated = proc_time;
        end
    end

    function [proc_time] = terminateParforProgress(print_progress_t)
        if(print_progress)
            percentDone = 100;
            msg = sprintf('%3.2f', percentDone);
        else
            msg = '';
            reverseStr = '';
        end
        if(nargin==1)
            print_progress = print_progress_t;
        end
        proc_time = toc(proc_start);
        if(print_progress)
            if(proc_time>86400)
                msg1 = sprintf('; Time taken = %3.1f days.\t\n',proc_time/86400);
            elseif(proc_time>3600)
                msg1 = sprintf('; Time taken = %3.1f hours.\t\n',proc_time/3600);
            elseif(proc_time>60)
                msg1 = sprintf('; Time taken = %3.1f minutes.\t\n',(proc_time/60));
            else
                msg1 = sprintf('; Time taken = %3.1f seconds.  \t\n',(proc_time));
            end
            msg = [msg,msg1];
            fprintf([reverseStr, msg]);
        end
    end

end

