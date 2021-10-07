function [bayesDetectorData] = get_subopt_bayes_detector_data_afhmm(params,max_valueFnIterations)
parforProgress = parallel.pool.DataQueue;
afterEach(parforProgress, @updateParforProgress);
percentDone = nan;
reverseStr = nan;
timeRemaining = nan;
proc_start = nan;
last_time_updated = nan;
msg1 = '';

y_num = params.y_num;
h_vec_num = params.h_vec_num;
P_Hk_vec_gHkn1_vec = params.P_Hk_vec_gHkn1_vec;
horizonsPerDay = params.horizonsPerDay;
P_XgHvec = params.P_XgHvec;
P_H1_vec_X1 = params.P_H1_vec_X1;

discountFactor = params.discountFactor;

possible_h_vec_idxs = params.possible_h_vec_idxs;
minLogParam = params.minLogParam;
lP_XgHvec = log(P_XgHvec);
lP_Hk_vec_gHkn1_vec = log(P_Hk_vec_gHkn1_vec);
lP_H1_vec_X1 = log(P_H1_vec_X1);

bayesDetectorData_filePrefix = 'cache/bayesDetectorData_afhmm_';
[bayesDetectorData_fileName,fileExists] = findFileName(params,bayesDetectorData_filePrefix,'params');
if(fileExists)
    fprintf(['\t\tDetection optimization skipped. Policy found in: ',bayesDetectorData_fileName,'\n']);
    load(bayesDetectorData_fileName,'bayesDetectorData');
else
    inst_reward = minLogParam*ones(h_vec_num,y_num,h_vec_num,horizonsPerDay);
    parfor hh_vec_kn1_idx = 1:h_vec_num
        lP_XgHvec_t = lP_XgHvec;
        lP_Hk_vec_gHkn1_vec_t = lP_Hk_vec_gHkn1_vec;
        possible_h_vec_idxs_t = possible_h_vec_idxs{hh_vec_kn1_idx};
        inst_reward_t = minLogParam*ones(h_vec_num,y_num,1,horizonsPerDay);
        for horizonIdx = 1:horizonsPerDay
            for y_k_idx = 1:y_num
                for hh_vec_k_idx = possible_h_vec_idxs_t
                    inst_reward_t(hh_vec_k_idx,y_k_idx,1,horizonIdx) = max(lP_XgHvec_t(y_k_idx,hh_vec_k_idx)+lP_Hk_vec_gHkn1_vec_t(hh_vec_k_idx,hh_vec_kn1_idx,horizonIdx),minLogParam);
                end
            end
        end
        inst_reward(:,:,hh_vec_kn1_idx,:) = inst_reward_t;
    end
        
    valueFunction_in1 = zeros(y_num,h_vec_num,horizonsPerDay);
    for iter_idx  = 1:max_valueFnIterations
        fprintf('\t\tPerforming value iteration step : %d - ',iter_idx);
        valueFunction_i = minLogParam*ones(y_num,h_vec_num,horizonsPerDay);
        det_strategy_i = zeros(y_num,h_vec_num,horizonsPerDay);
        
        discounted_approxValueFn_in1 = discountFactor*valueFunction_in1;
        exp_discounted_approxValueFn_in1 = minLogParam*ones(h_vec_num,horizonsPerDay);
        initializeParforProgress();
        incPercent = (1/h_vec_num)*50;
        parfor hh_vec_k_idx = 1:h_vec_num
            possible_h_vec_idxs_t = possible_h_vec_idxs{hh_vec_k_idx};
            P_XgHvec_t = P_XgHvec;
            P_Hk_vec_gHkn1_vec_t = P_Hk_vec_gHkn1_vec;
            discounted_approxValueFn_in1_t = discounted_approxValueFn_in1;
            exp_discounted_approxValueFn_in1_t = minLogParam*ones(1,horizonsPerDay);
            for horizonIdx = 1:horizonsPerDay
                temp = 0;
                for h_vec_kp1_idx = possible_h_vec_idxs_t
                    temp = temp + P_Hk_vec_gHkn1_vec_t(h_vec_kp1_idx,hh_vec_k_idx,horizonIdx)*...
                        P_XgHvec_t(:,h_vec_kp1_idx)'*discounted_approxValueFn_in1_t(:,hh_vec_k_idx,horizonIdx);
                end
                exp_discounted_approxValueFn_in1_t(horizonIdx) = temp;
            end
            exp_discounted_approxValueFn_in1(hh_vec_k_idx,:) = exp_discounted_approxValueFn_in1_t;
            send(parforProgress, incPercent);
        end        
        parfor hh_vec_kn1_idx = 1:h_vec_num
            inst_reward_t = inst_reward(:,:,hh_vec_kn1_idx,:);
            exp_discounted_approxValueFn_in1_t = exp_discounted_approxValueFn_in1;
                        
            det_strategy_i_t = zeros(y_num,1,horizonsPerDay);
            valueFunction_i_t = minLogParam*ones(y_num,1,horizonsPerDay);
            
            for horizonIdx = 1:horizonsPerDay
                for y_k_idx = 1:y_num
                    value_func_t = inst_reward_t(:,y_k_idx,1,horizonIdx) + exp_discounted_approxValueFn_in1_t(:,horizonIdx);
                    [max_val,opt_hh_vec_k_idx] = max(value_func_t);
                    valueFunction_i_t(y_k_idx,1,horizonIdx) = max_val;
                    det_strategy_i_t(y_k_idx,1,horizonIdx) = opt_hh_vec_k_idx;
                end
            end
            
            valueFunction_i(:,hh_vec_kn1_idx,:) = valueFunction_i_t;
            det_strategy_i(:,hh_vec_kn1_idx,:) = det_strategy_i_t;
            send(parforProgress, incPercent);
        end
        
        valueFunction_diff = valueFunction_i(:)-valueFunction_in1(:);
        valueFunction_diff = abs(valueFunction_diff(~isinf(valueFunction_diff)));        
        max_val_inc = max(valueFunction_diff);
        terminateParforProgress(max_val_inc);
        
        bayesDetectorData = struct;
        bayesDetectorData.det_strategy = det_strategy_i;
        bayesDetectorData.valueFunction = valueFunction_i;
        bayesDetectorData.valueFunction_in1 = valueFunction_in1;
        bayesDetectorData.max_val_inc = max_val_inc;
        bayesDetectorData.iter_idx = iter_idx;
        
        save(bayesDetectorData_fileName,'bayesDetectorData','params')
        if(max_val_inc<=params.min_max_val_inc || iter_idx==max_valueFnIterations)    
            fprintf('\t\tPerforming initialization step optimization : - ');
            valueFunction_in1 = bayesDetectorData.valueFunction_in1;
            det_strategy_1 = zeros(y_num,horizonsPerDay);
            
            discounted_approxValueFn_in1 = discountFactor*valueFunction_in1;
            exp_discounted_approxValueFn_in1 = minLogParam*ones(h_vec_num,horizonsPerDay);
            initializeParforProgress();
            incPercent = (1/h_vec_num)*50;
            parfor hh_vec_k_idx = 1:h_vec_num
                possible_h_vec_idxs_t = possible_h_vec_idxs{hh_vec_k_idx};
                P_XgHvec_t = P_XgHvec;
                P_Hk_vec_gHkn1_vec_t = P_Hk_vec_gHkn1_vec;
                discounted_approxValueFn_in1_t = discounted_approxValueFn_in1;
                exp_discounted_approxValueFn_in1_t = minLogParam*ones(1,horizonsPerDay);
                for horizonIdx = 1:horizonsPerDay
                    temp = 0;
                    for h_vec_kp1_idx = possible_h_vec_idxs_t
                        temp = temp + P_Hk_vec_gHkn1_vec_t(h_vec_kp1_idx,hh_vec_k_idx,horizonIdx)*...
                            P_XgHvec_t(:,h_vec_kp1_idx)'*discounted_approxValueFn_in1_t(:,hh_vec_k_idx,horizonIdx);
                    end
                    exp_discounted_approxValueFn_in1_t(horizonIdx) = temp;
                end
                exp_discounted_approxValueFn_in1(hh_vec_k_idx,:) = exp_discounted_approxValueFn_in1_t;
                send(parforProgress, incPercent);
            end
            
            inst_reward = max(lP_H1_vec_X1,minLogParam);            
            incPercent = (1/y_num)*50;
            parfor y_k_idx = 1:y_num
                inst_reward_t = inst_reward(:,y_k_idx,:);
                exp_discounted_approxValueFn_in1_t = exp_discounted_approxValueFn_in1;
                
                det_strategy_i_t = zeros(1,horizonsPerDay);
                for horizonIdx = 1:horizonsPerDay
                    value_func_t = inst_reward_t(:,1,horizonIdx) + exp_discounted_approxValueFn_in1_t(:,horizonIdx);
                    [~,opt_hh_vec_k_idx] = max(value_func_t);
                    det_strategy_i_t(horizonIdx) = opt_hh_vec_k_idx;
                end
                det_strategy_1(y_k_idx,:) = det_strategy_i_t;
                send(parforProgress, incPercent);
            end
            terminateParforProgress;
            
            bayesDetectorData.det_strategy_1 = det_strategy_1;
            save(bayesDetectorData_fileName,'bayesDetectorData','params')   
            break;
        end
        valueFunction_in1 = valueFunction_i;
    end
    fprintf(['\t\tOptimization complete. Policy saved in: ',bayesDetectorData_fileName,'\n']);
end

valueFnIterationsDone = bayesDetectorData.iter_idx;
if(max_valueFnIterations>valueFnIterationsDone)
    if(bayesDetectorData.max_val_inc>params.min_max_val_inc)
        inst_reward = minLogParam*ones(h_vec_num,y_num,h_vec_num,horizonsPerDay);
        parfor hh_vec_kn1_idx = 1:h_vec_num
            lP_XgHvec_t = lP_XgHvec;
            lP_Hk_vec_gHkn1_vec_t = lP_Hk_vec_gHkn1_vec;
            possible_h_vec_idxs_t = possible_h_vec_idxs{hh_vec_kn1_idx};
            inst_reward_t = minLogParam*ones(h_vec_num,y_num,1,horizonsPerDay);
            for horizonIdx = 1:horizonsPerDay
                for y_k_idx = 1:y_num
                    for hh_vec_k_idx = possible_h_vec_idxs_t
                        inst_reward_t(hh_vec_k_idx,y_k_idx,1,horizonIdx) = max(lP_XgHvec_t(y_k_idx,hh_vec_k_idx)+lP_Hk_vec_gHkn1_vec_t(hh_vec_k_idx,hh_vec_kn1_idx,horizonIdx),minLogParam);
                    end
                end
            end
            inst_reward(:,:,hh_vec_kn1_idx,:) = inst_reward_t;
        end
        
        valueFunction_in1 = bayesDetectorData.valueFunction_in1;        
        for iter_idx  = valueFnIterationsDone+1:max_valueFnIterations
            fprintf('\t\tPerforming value iteration step : %d - ',iter_idx);
            valueFunction_i = minLogParam*ones(y_num,h_vec_num,horizonsPerDay);
            det_strategy_i = zeros(y_num,h_vec_num,horizonsPerDay);
            
            discounted_approxValueFn_in1 = discountFactor*valueFunction_in1;
            exp_discounted_approxValueFn_in1 = minLogParam*ones(h_vec_num,horizonsPerDay);
            initializeParforProgress();
            incPercent = (1/h_vec_num)*50;
            parfor hh_vec_k_idx = 1:h_vec_num
                possible_h_vec_idxs_t = possible_h_vec_idxs{hh_vec_k_idx};
                P_XgHvec_t = P_XgHvec;
                P_Hk_vec_gHkn1_vec_t = P_Hk_vec_gHkn1_vec;
                discounted_approxValueFn_in1_t = discounted_approxValueFn_in1;
                exp_discounted_approxValueFn_in1_t = minLogParam*ones(1,horizonsPerDay);
                for horizonIdx = 1:horizonsPerDay
                    temp = 0;
                    for h_vec_kp1_idx = possible_h_vec_idxs_t
                        temp = temp + P_Hk_vec_gHkn1_vec_t(h_vec_kp1_idx,hh_vec_k_idx,horizonIdx)*...
                            P_XgHvec_t(:,h_vec_kp1_idx)'*discounted_approxValueFn_in1_t(:,hh_vec_k_idx,horizonIdx);
                    end
                    exp_discounted_approxValueFn_in1_t(horizonIdx) = temp;
                end
                exp_discounted_approxValueFn_in1(hh_vec_k_idx,:) = exp_discounted_approxValueFn_in1_t;
                send(parforProgress, incPercent);
            end
            parfor hh_vec_kn1_idx = 1:h_vec_num
                inst_reward_t = inst_reward(:,:,hh_vec_kn1_idx,:);
                exp_discounted_approxValueFn_in1_t = exp_discounted_approxValueFn_in1;
                
                det_strategy_i_t = zeros(y_num,1,horizonsPerDay);
                valueFunction_i_t = minLogParam*ones(y_num,1,horizonsPerDay);
                
                for horizonIdx = 1:horizonsPerDay
                    for y_k_idx = 1:y_num
                        value_func_t = inst_reward_t(:,y_k_idx,1,horizonIdx) + exp_discounted_approxValueFn_in1_t(:,horizonIdx);
                        [max_val,opt_hh_vec_k_idx] = max(value_func_t);
                        valueFunction_i_t(y_k_idx,1,horizonIdx) = max_val;
                        det_strategy_i_t(y_k_idx,1,horizonIdx) = opt_hh_vec_k_idx;
                    end
                end
                
                valueFunction_i(:,hh_vec_kn1_idx,:) = valueFunction_i_t;
                det_strategy_i(:,hh_vec_kn1_idx,:) = det_strategy_i_t;
                send(parforProgress, incPercent);
            end
                        
            valueFunction_diff = valueFunction_i(:)-valueFunction_in1(:);
            valueFunction_diff = abs(valueFunction_diff(~isinf(valueFunction_diff)));
            max_val_inc = max(valueFunction_diff);
            terminateParforProgress(max_val_inc);
            
            bayesDetectorData = struct;
            bayesDetectorData.det_strategy = det_strategy_i;
            bayesDetectorData.valueFunction = valueFunction_i;
            bayesDetectorData.valueFunction_in1 = valueFunction_in1;
            bayesDetectorData.max_val_inc = max_val_inc;
            bayesDetectorData.iter_idx = iter_idx;
            
            save(bayesDetectorData_fileName,'bayesDetectorData','params')
            if(max_val_inc<=params.min_max_val_inc || iter_idx==max_valueFnIterations)
                fprintf('\t\tPerforming initialization step optimization : - ');
                valueFunction_in1 = bayesDetectorData.valueFunction_in1;
                det_strategy_1 = zeros(y_num,horizonsPerDay);
                
                discounted_approxValueFn_in1 = discountFactor*valueFunction_in1;
                exp_discounted_approxValueFn_in1 = minLogParam*ones(h_vec_num,horizonsPerDay);
                initializeParforProgress();
                incPercent = (1/h_vec_num)*50;
                parfor hh_vec_k_idx = 1:h_vec_num
                    possible_h_vec_idxs_t = possible_h_vec_idxs{hh_vec_k_idx};
                    P_XgHvec_t = P_XgHvec;
                    P_Hk_vec_gHkn1_vec_t = P_Hk_vec_gHkn1_vec;
                    discounted_approxValueFn_in1_t = discounted_approxValueFn_in1;
                    exp_discounted_approxValueFn_in1_t = minLogParam*ones(1,horizonsPerDay);
                    for horizonIdx = 1:horizonsPerDay
                        temp = 0;
                        for h_vec_kp1_idx = possible_h_vec_idxs_t
                            temp = temp + P_Hk_vec_gHkn1_vec_t(h_vec_kp1_idx,hh_vec_k_idx,horizonIdx)*...
                                P_XgHvec_t(:,h_vec_kp1_idx)'*discounted_approxValueFn_in1_t(:,hh_vec_k_idx,horizonIdx);
                        end
                        exp_discounted_approxValueFn_in1_t(horizonIdx) = temp;
                    end
                    exp_discounted_approxValueFn_in1(hh_vec_k_idx,:) = exp_discounted_approxValueFn_in1_t;
                    send(parforProgress, incPercent);
                end
                
                inst_reward = max(lP_H1_vec_X1,minLogParam);
                incPercent = (1/y_num)*50;
                parfor y_k_idx = 1:y_num
                    inst_reward_t = inst_reward(:,y_k_idx,:);
                    exp_discounted_approxValueFn_in1_t = exp_discounted_approxValueFn_in1;
                    
                    det_strategy_i_t = zeros(1,horizonsPerDay);
                    for horizonIdx = 1:horizonsPerDay
                        value_func_t = inst_reward_t(:,1,horizonIdx) + exp_discounted_approxValueFn_in1_t(:,horizonIdx);
                        [~,opt_hh_vec_k_idx] = max(value_func_t);
                        det_strategy_i_t(horizonIdx) = opt_hh_vec_k_idx;
                    end
                    det_strategy_1(y_k_idx,:) = det_strategy_i_t;
                    send(parforProgress, incPercent);
                end
                terminateParforProgress;
                
                bayesDetectorData.det_strategy_1 = det_strategy_1;
                save(bayesDetectorData_fileName,'bayesDetectorData','params')
                break;
            end
            valueFunction_in1 = valueFunction_i;
        end
        fprintf(['\t\tOptimization complete. Policy saved in: ',bayesDetectorData_fileName,'\n']);
    end
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


