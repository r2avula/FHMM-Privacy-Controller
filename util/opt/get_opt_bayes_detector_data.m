function [bayesDetectorData] = get_opt_bayes_detector_data(bayesDetectorParams)
parforProgress = parallel.pool.DataQueue;
afterEach(parforProgress, @updateParforProgress);
percentDone = nan;
reverseStr = nan;
timeRemaining = nan;
proc_start = nan;
last_time_updated = nan;
msg1 = '';

beliefSpacePrecision = bayesDetectorParams.beliefSpacePrecision;
x_num = bayesDetectorParams.x_num;
h_vec_num = bayesDetectorParams.h_vec_num;
P_XkgXkn1HkHkn1 = bayesDetectorParams.P_XkgXkn1HkHkn1;
P_Hk_vec_gHkn1_vec = bayesDetectorParams.P_Hk_vec_gHkn1_vec;
horizonsPerDay = bayesDetectorParams.horizonsPerDay;
minLikelihoodFilter = bayesDetectorParams.minLikelihoodFilter;
P_H1_vec_X1 = bayesDetectorParams.P_H1_vec_X1;

bayesDetectorData_filePrefix = 'cache/bayesDetectorData_';
[bayesDetectorData_fileName,fileExists] = findFileName(bayesDetectorParams,bayesDetectorData_filePrefix,'bayesDetectorParams');
if(fileExists)
    load(bayesDetectorData_fileName,'bayesDetectorData');
else    
    % Belief space discretization
    prob_int_sum = floor(1/beliefSpacePrecision);
    prob_temp1 = nchoosek(1:(prob_int_sum+h_vec_num-1), h_vec_num-1);
    prob_ndividers = size(prob_temp1, 1);
    prob_temp2 = cat(2, zeros(prob_ndividers, 1), prob_temp1, (prob_int_sum+h_vec_num)*ones(prob_ndividers, 1));
    h_vec_pdbs = beliefSpacePrecision*(diff(prob_temp2, 1, 2) - 1)';
    h_vec_pdbs_num = size(h_vec_pdbs,2);
    h_vec_pdbs_T = h_vec_pdbs';
    
    a_roundOffBelief_dbs_fn = @(x)roundOffBelief(x,beliefSpacePrecision,h_vec_pdbs_T);
    
    belief_transf = zeros(h_vec_num,h_vec_num,x_num,x_num,horizonsPerDay);
    for horizonIdx = 1:horizonsPerDay
        for x_kn1_idx = 1:x_num
            for x_k_idx = 1:x_num
                belief_transf(:,:,x_k_idx,x_kn1_idx,horizonIdx) =  reshape(P_XkgXkn1HkHkn1(x_k_idx,x_kn1_idx,:,:),h_vec_num,h_vec_num).*P_Hk_vec_gHkn1_vec(:,:,horizonIdx);
            end
        end
    end
    
    [~,P_H_vec_default_idx] = a_roundOffBelief_dbs_fn(ones(h_vec_num,1)/h_vec_num);
        
    HhIdxs_given_P_h_vec_idx = zeros(1,h_vec_pdbs_num);
    adv_belief_transition_map = P_H_vec_default_idx*ones(x_num,x_num,h_vec_pdbs_num,horizonsPerDay);
    fprintf('\t\t\tComputing adversarial detection map : ');
    initializeParforProgress();
    incPercent = (1/h_vec_pdbs_num)*100;
    parfor P_h_vec_idx = 1:h_vec_pdbs_num
        belief_transf_t = belief_transf;
        
        P_H_vec = h_vec_pdbs(:,P_h_vec_idx);
        [~,HhIdxs_given_P_h_vec_idx(P_h_vec_idx)] = max(P_H_vec); 
        
        for horizonIdx = 1:horizonsPerDay
            for y_kn1_idx_obs = 1:x_num
                for y_k_idx = 1:x_num
                    belief_k = belief_transf_t(:,:,y_k_idx,y_kn1_idx_obs,horizonIdx)*P_H_vec;
                    sum_belief_k = sum(belief_k);
                    if(sum_belief_k>minLikelihoodFilter)
                        [~,adv_belief_k_idx] = a_roundOffBelief_dbs_fn(belief_k/sum_belief_k)
                        adv_belief_transition_map(y_k_idx,y_kn1_idx_obs,P_h_vec_idx,horizonIdx) = adv_belief_k_idx;
                    end
                end
            end
        end
        send(parforProgress, incPercent);
    end
    terminateParforProgress;
    
    P_H1_vec_idxs = P_H_vec_default_idx*ones(x_num,horizonsPerDay);
    for horizonIdx = 1:horizonsPerDay
        for x_idx = 1:x_num
            P_H1_vec_t = P_H1_vec_X1(:,x_idx,horizonIdx);
            P_H1_vec_sum_t = sum(P_H1_vec_t);
            if(P_H1_vec_sum_t>minLikelihoodFilter)
                 [~,P_H1_vec_idxs(x_idx,horizonIdx)] = a_roundOffBelief_dbs_fn(P_H1_vec_t/P_H1_vec_sum_t);
            end
        end
    end
    
    bayesDetectorData = struct;
    bayesDetectorData.P_H1_vec_idxs = P_H1_vec_idxs;
    bayesDetectorData.P_H_vec_default_idx = P_H_vec_default_idx;
    bayesDetectorData.HhIdxs_given_P_h_vec_idx = HhIdxs_given_P_h_vec_idx;
    bayesDetectorData.adv_belief_transition_map = adv_belief_transition_map;
    bayesDetectorData.h_vec_pdbs_T = h_vec_pdbs_T;
    bayesDetectorData.h_vec_pdbs_num = h_vec_pdbs_num;
    bayesDetectorData.belief_transf = belief_transf;
    
    save(bayesDetectorData_fileName,'bayesDetectorData','bayesDetectorParams');
end

    function initializeParforProgress(noPrint)
        if(nargin<1)
            noPrint = false;
        end
        percentDone = 0;
        timeRemaining = nan;
        proc_start = tic;
        last_time_updated = -inf;
        if(noPrint)
            reverseStr = [];
        else
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
        msg = sprintf('%3.2f', percentDone);
        proc_time = toc(proc_start);
        if(proc_time-last_time_updated >1)
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
            last_time_updated = proc_time;
        end
        msg = [msg,msg1];
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
    end

    function [proc_time] = terminateParforProgress(max_val_inc)
        percentDone = 100;
        msg = sprintf('%3.2f', percentDone);
        proc_time = toc(proc_start);
        if(nargin==1)
            if(proc_time>86400)
                msg1 = sprintf('; Time taken = %3.1f days; Max. value increment : %f.\t\n',proc_time/86400,max_val_inc);
            elseif(proc_time>3600)
                msg1 = sprintf('; Time taken = %3.1f hours; Max. value increment : %f.\t\n',proc_time/3600,max_val_inc);
            elseif(proc_time>60)
                msg1 = sprintf('; Time taken = %3.1f minutes; Max. value increment : %f.\t\n',(proc_time/60),max_val_inc);
            else
                msg1 = sprintf('; Time taken = %3.1f seconds; Max. value increment : %f.\t\n',(proc_time),max_val_inc);
            end
        else
            if(proc_time>86400)
                msg1 = sprintf('; Time taken = %3.1f days.\t\n',proc_time/86400);
            elseif(proc_time>3600)
                msg1 = sprintf('; Time taken = %3.1f hours.\t\n',proc_time/3600);
            elseif(proc_time>60)
                msg1 = sprintf('; Time taken = %3.1f minutes.\t\n',(proc_time/60));
            else
                msg1 = sprintf('; Time taken = %3.1f seconds.\t\n',(proc_time));
            end
        end
        msg = [msg,msg1];
        fprintf([reverseStr, msg]);
    end
end


