function [policy] = get_opt_hyp_aware_policy(params,bayesDetectorData,max_valueFnIterations)
parforProgress = parallel.pool.DataQueue;
afterEach(parforProgress, @updateParforProgress);
percentDone = nan;
reverseStr = nan;
timeRemaining = nan;
proc_start = nan;
last_time_updated = nan;
msg1 = '';

x_num = params.x_num;
z_num = params.z_num;
y_num = params.y_num;
h_vec_num = params.h_vec_num;
P_XkgXkn1HkHkn1 = params.P_XkgXkn1HkHkn1;
P_Hk_vec_gHkn1_vec = params.P_Hk_vec_gHkn1_vec;
horizonsPerDay = params.horizonsPerDay;
P_Zp1gZD = params.P_Zp1gZD;
d_offset = params.d_offset;
discountFactor = params.discountFactor;
C_HgHh_privacy_vecs = params.C_HgHh_privacy_vecs;
possible_h_vec_idxs = params.possible_h_vec_idxs;
maxCostParam = params.maxCostParam;
valid_y_idxs = params.valid_y_idxs;
P_H1_vec_idxs = bayesDetectorData.P_H1_vec_idxs;
HhIdxs_given_P_h_vec_idx = bayesDetectorData.HhIdxs_given_P_h_vec_idx;
adv_belief_transition_map = bayesDetectorData.adv_belief_transition_map;
h_vec_pdbs_num = bayesDetectorData.h_vec_pdbs_num;

params_t = struct;
params_t.x_num = x_num;
params_t.z_num = z_num;
params_t.y_num = y_num;
params_t.h_vec_num = h_vec_num;
params_t.P_XkgXkn1HkHkn1 = P_XkgXkn1HkHkn1;
params_t.P_Hk_vec_gHkn1_vec = P_Hk_vec_gHkn1_vec;
params_t.horizonsPerDay = horizonsPerDay;
params_t.P_Zp1gZD = P_Zp1gZD;
params_t.d_offset = d_offset;
params_t.discountFactor = discountFactor;
params_t.C_HgHh_privacy_vecs = C_HgHh_privacy_vecs;
params_t.possible_h_vec_idxs = possible_h_vec_idxs;
params_t.maxCostParam = maxCostParam;
params_t.valid_y_idxs = valid_y_idxs;
params_t.P_H1_vec_idxs = P_H1_vec_idxs;
params_t.HhIdxs_given_P_h_vec_idx = HhIdxs_given_P_h_vec_idx;
params_t.adv_belief_transition_map = adv_belief_transition_map;
params_t.h_vec_pdbs_num = h_vec_pdbs_num;

stat_policy_fileNamePrefix = 'cache/stat_policy_ha_bayesian_control_';
[policy_fileName,fileExists] = findFileName(params_t,stat_policy_fileNamePrefix,'params_t');
if(fileExists)
    fprintf(['\t\tController optimization skipped. Policy found in: ',policy_fileName,'\n']);
    load(policy_fileName,'policy');
else    
    valueFunction_in1 = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay,h_vec_pdbs_num);    
    for iter_idx  = 1:max_valueFnIterations
        fprintf('\t\tPerforming value iteration step : %d - ',iter_idx);
        valueFunction_i = maxCostParam*ones(x_num,z_num,y_num,h_vec_num,horizonsPerDay,h_vec_pdbs_num);
        emu_strategy_i = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay,h_vec_pdbs_num);
        
        discounted_approxValueFn_in1 = discountFactor*valueFunction_in1;
        exp_discounted_approxValueFn_in1 = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay,h_vec_pdbs_num);
        initializeParforProgress();
        incPercent = (1/h_vec_pdbs_num)*50;
        parfor P_h_vec_k_idx = 1:h_vec_pdbs_num
            P_Zp1gZD_t = P_Zp1gZD;
            valid_y_idxs_t = valid_y_idxs;
            possible_h_vec_idxs_t = possible_h_vec_idxs;
            P_XkgXkn1HkHkn1_t = P_XkgXkn1HkHkn1;
            P_Hk_vec_gHkn1_vec_t = P_Hk_vec_gHkn1_vec;
            discounted_approxValueFn_in1_t = discounted_approxValueFn_in1(:,:,:,:,:,P_h_vec_k_idx);
            exp_discounted_approxValueFn_in1_t = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay);
            for x_k_idx = 1:x_num
                for z_kn1_idx = 1:z_num
                    for y_k_idx = valid_y_idxs_t{x_k_idx,z_kn1_idx}
                        d_k_idx = y_k_idx - x_k_idx - d_offset;
                        z_kp1_idx_prob = P_Zp1gZD_t(:,z_kn1_idx,d_k_idx);
                        for horizonIdx = 1:horizonsPerDay
                            for h_vec_k_idx = 1:h_vec_num
                                temp = 0;
                                for h_vec_kp1_idx = possible_h_vec_idxs_t{h_vec_k_idx}
                                    temp = temp + P_Hk_vec_gHkn1_vec_t(h_vec_kp1_idx,h_vec_k_idx,horizonIdx)*...
                                        P_XkgXkn1HkHkn1_t(:,x_k_idx,h_vec_kp1_idx,h_vec_k_idx)'*discounted_approxValueFn_in1_t(:,:,y_k_idx,h_vec_kp1_idx,horizonIdx)*z_kp1_idx_prob;
                                end
                                exp_discounted_approxValueFn_in1_t(x_k_idx,z_kn1_idx,y_k_idx,h_vec_k_idx,horizonIdx) = temp;
                            end
                        end
                    end
                end
            end
            exp_discounted_approxValueFn_in1(:,:,:,:,:,P_h_vec_k_idx) = exp_discounted_approxValueFn_in1_t;
            send(parforProgress, incPercent);
        end
        parfor P_h_vec_kn1_idx = 1:h_vec_pdbs_num
            C_HgHh_privacy_vecs_t = C_HgHh_privacy_vecs;
            HhIdxs_given_P_h_vec_idx_t = HhIdxs_given_P_h_vec_idx;
            adv_belief_transition_map_t = adv_belief_transition_map(:,:,P_h_vec_kn1_idx,:);
            valid_y_idxs_t = valid_y_idxs;
            exp_discounted_approxValueFn_in1_t = exp_discounted_approxValueFn_in1;
            
            emu_strategy_i_t = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay);
            valueFunction_i_t = maxCostParam*ones(x_num,z_num,y_num,h_vec_num,horizonsPerDay);
            
            for horizonIdx = 1:horizonsPerDay
                for z_kn1_idx = 1:z_num
                    for x_k_idx = 1:x_num
                        for h_vec_k_idx = 1:h_vec_num
                            for y_kn1_idx = 1:y_num
                                value_func_t = maxCostParam*ones(1,y_num);
                                for y_k_idx = valid_y_idxs_t{x_k_idx,z_kn1_idx}
                                    P_h_vec_k_idx = adv_belief_transition_map_t(y_k_idx,y_kn1_idx,1,horizonIdx);
                                    value_func_t(y_k_idx) = C_HgHh_privacy_vecs_t(h_vec_k_idx,HhIdxs_given_P_h_vec_idx_t(P_h_vec_k_idx)) + ...
                                        exp_discounted_approxValueFn_in1_t(x_k_idx,z_kn1_idx,y_k_idx,h_vec_k_idx,horizonIdx,P_h_vec_k_idx);
                                end
                                
                                [min_val,opt_y_idx] = min(value_func_t);
                                valueFunction_i_t(x_k_idx,z_kn1_idx,y_kn1_idx,h_vec_k_idx,horizonIdx) = min_val;
                                emu_strategy_i_t(x_k_idx,z_kn1_idx,y_kn1_idx,h_vec_k_idx,horizonIdx) = opt_y_idx;
                            end
                        end
                    end
                end
            end
            
            valueFunction_i(:,:,:,:,:,P_h_vec_kn1_idx) = valueFunction_i_t;
            emu_strategy_i(:,:,:,:,:,P_h_vec_kn1_idx) = emu_strategy_i_t;
            send(parforProgress, incPercent);
        end
        max_val_inc = max(valueFunction_i(:)-valueFunction_in1(:));
        terminateParforProgress(max_val_inc);
        
        policy = struct;
        policy.emu_strategy = emu_strategy_i;
        policy.valueFunction = valueFunction_i;
        policy.valueFunction_in1 = valueFunction_in1;
        policy.max_val_inc = max_val_inc;
        policy.iter_idx = iter_idx;
        
        valueFunction_in1 = valueFunction_i;
        save(policy_fileName,'policy','params_t')
        if(max_val_inc<=params.min_max_val_inc || iter_idx ==max_valueFnIterations)
            fprintf('\t\tPerforming initialization step optimization : - ');
            valueFunction_in1 = policy.valueFunction_in1;
            emu_strategy_1 = zeros(x_num,z_num,h_vec_num,horizonsPerDay);
            
            discounted_approxValueFn_in1 = discountFactor*valueFunction_in1;
            exp_discounted_approxValueFn_in1 = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay);
            initializeParforProgress();
            incPercent = (1/x_num)*50;
            parfor x_k_idx = 1:x_num
                P_Zp1gZD_t = P_Zp1gZD;
                valid_y_idxs_t = valid_y_idxs(x_k_idx,:);
                possible_h_vec_idxs_t = possible_h_vec_idxs;
                P_XkgXkn1HkHkn1_t = P_XkgXkn1HkHkn1;
                P_Hk_vec_gHkn1_vec_t = P_Hk_vec_gHkn1_vec;
                discounted_approxValueFn_in1_t = discounted_approxValueFn_in1;
                P_H1_vec_idxs_t = P_H1_vec_idxs;
                
                exp_discounted_approxValueFn_in1_t = zeros(1,z_num,y_num,h_vec_num,horizonsPerDay);
                for z_kn1_idx = 1:z_num
                    for y_k_idx = valid_y_idxs_t{z_kn1_idx}
                        d_k_idx = y_k_idx - x_k_idx - d_offset;
                        z_kp1_idx_prob = P_Zp1gZD_t(:,z_kn1_idx,d_k_idx);
                        for horizonIdx = 1:horizonsPerDay
                            P_h_vec_k_idx = P_H1_vec_idxs_t(y_k_idx,horizonIdx);
                            for h_vec_k_idx = 1:h_vec_num
                                temp = 0;
                                for h_vec_kp1_idx = possible_h_vec_idxs_t{h_vec_k_idx}
                                    temp = temp + P_Hk_vec_gHkn1_vec_t(h_vec_kp1_idx,h_vec_k_idx,horizonIdx)*...
                                        P_XkgXkn1HkHkn1_t(:,x_k_idx,h_vec_kp1_idx,h_vec_k_idx)'*discounted_approxValueFn_in1_t(:,:,y_k_idx,h_vec_kp1_idx,horizonIdx,P_h_vec_k_idx)*z_kp1_idx_prob;
                                end
                                exp_discounted_approxValueFn_in1_t(1,z_kn1_idx,y_k_idx,h_vec_k_idx,horizonIdx) = temp;
                            end
                        end
                    end
                end
                
                exp_discounted_approxValueFn_in1(x_k_idx,:,:,:,:) = exp_discounted_approxValueFn_in1_t;
                send(parforProgress, incPercent);
            end            
            parfor x_k_idx = 1:x_num
                C_HgHh_privacy_vecs_t = C_HgHh_privacy_vecs;
                HhIdxs_given_P_h_vec_idx_t = HhIdxs_given_P_h_vec_idx;
                P_H1_vec_idxs_t = P_H1_vec_idxs;
                valid_y_idxs_t = valid_y_idxs(x_k_idx,:);
                exp_discounted_approxValueFn_in1_t = exp_discounted_approxValueFn_in1(x_k_idx,:,:,:,:);
                
                emu_strategy_1_t = zeros(1,z_num,h_vec_num,horizonsPerDay);
                for horizonIdx = 1:horizonsPerDay
                    for z_kn1_idx = 1:z_num
                        for h_vec_k_idx = 1:h_vec_num
                            value_func_t = maxCostParam*ones(1,y_num);
                            for y_k_idx = valid_y_idxs_t{z_kn1_idx}
                                P_h_vec_k_idx = P_H1_vec_idxs_t(y_k_idx,horizonIdx);
                                value_func_t(y_k_idx) = C_HgHh_privacy_vecs_t(h_vec_k_idx,HhIdxs_given_P_h_vec_idx_t(P_h_vec_k_idx)) + ...
                                    exp_discounted_approxValueFn_in1_t(1,z_kn1_idx,y_k_idx,h_vec_k_idx,horizonIdx);
                            end
                            
                            [~,opt_y_idx] = min(value_func_t);
                            emu_strategy_1_t(1,z_kn1_idx,h_vec_k_idx,horizonIdx) = opt_y_idx;
                        end
                    end
                end
                
                emu_strategy_1(x_k_idx,:,:,:) = emu_strategy_1_t;
                send(parforProgress, incPercent);
            end
            terminateParforProgress;            
            policy.emu_strategy_1 = emu_strategy_1;
            save(policy_fileName,'policy','params_t')            
            break;
        end
    end
    fprintf(['\t\tOptimization complete. Policy saved in: ',policy_fileName,'\n']);
end

valueFnIterationsDone = policy.iter_idx;
if(max_valueFnIterations>valueFnIterationsDone)
    valueFunction_in1 = policy.valueFunction_in1;
    if(policy.max_val_inc>params.min_max_val_inc)
        for iter_idx  = valueFnIterationsDone+1:max_valueFnIterations
            fprintf('\t\tPerforming value iteration step : %d - ',iter_idx);
            valueFunction_i = maxCostParam*ones(x_num,z_num,y_num,h_vec_num,horizonsPerDay,h_vec_pdbs_num);
            emu_strategy_i = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay,h_vec_pdbs_num);
            
            discounted_approxValueFn_in1 = discountFactor*valueFunction_in1;
            exp_discounted_approxValueFn_in1 = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay,h_vec_pdbs_num);
            initializeParforProgress();
            incPercent = (1/h_vec_pdbs_num)*50;
            parfor P_h_vec_k_idx = 1:h_vec_pdbs_num
                P_Zp1gZD_t = P_Zp1gZD;
                valid_y_idxs_t = valid_y_idxs;
                possible_h_vec_idxs_t = possible_h_vec_idxs;
                P_XkgXkn1HkHkn1_t = P_XkgXkn1HkHkn1;
                P_Hk_vec_gHkn1_vec_t = P_Hk_vec_gHkn1_vec;
                discounted_approxValueFn_in1_t = discounted_approxValueFn_in1(:,:,:,:,:,P_h_vec_k_idx);
                exp_discounted_approxValueFn_in1_t = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay);
                for x_k_idx = 1:x_num
                    for z_kn1_idx = 1:z_num
                        for y_k_idx = valid_y_idxs_t{x_k_idx,z_kn1_idx}
                            d_k_idx = y_k_idx - x_k_idx - d_offset;
                            z_kp1_idx_prob = P_Zp1gZD_t(:,z_kn1_idx,d_k_idx);
                            for horizonIdx = 1:horizonsPerDay
                                for h_vec_k_idx = 1:h_vec_num
                                    temp = 0;
                                    for h_vec_kp1_idx = possible_h_vec_idxs_t{h_vec_k_idx}
                                        temp = temp + P_Hk_vec_gHkn1_vec_t(h_vec_kp1_idx,h_vec_k_idx,horizonIdx)*...
                                            P_XkgXkn1HkHkn1_t(:,x_k_idx,h_vec_kp1_idx,h_vec_k_idx)'*discounted_approxValueFn_in1_t(:,:,y_k_idx,h_vec_kp1_idx,horizonIdx)*z_kp1_idx_prob;
                                    end
                                    exp_discounted_approxValueFn_in1_t(x_k_idx,z_kn1_idx,y_k_idx,h_vec_k_idx,horizonIdx) = temp;
                                end
                            end
                        end
                    end
                end
                exp_discounted_approxValueFn_in1(:,:,:,:,:,P_h_vec_k_idx) = exp_discounted_approxValueFn_in1_t;
                send(parforProgress, incPercent);
            end
            parfor P_h_vec_kn1_idx = 1:h_vec_pdbs_num
                C_HgHh_privacy_vecs_t = C_HgHh_privacy_vecs;
                HhIdxs_given_P_h_vec_idx_t = HhIdxs_given_P_h_vec_idx;
                adv_belief_transition_map_t = adv_belief_transition_map(:,:,P_h_vec_kn1_idx,:);
                valid_y_idxs_t = valid_y_idxs;
                exp_discounted_approxValueFn_in1_t = exp_discounted_approxValueFn_in1;
                
                emu_strategy_i_t = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay);
                valueFunction_i_t = maxCostParam*ones(x_num,z_num,y_num,h_vec_num,horizonsPerDay);
                
                for horizonIdx = 1:horizonsPerDay
                    for z_kn1_idx = 1:z_num
                        for x_k_idx = 1:x_num
                            for h_vec_k_idx = 1:h_vec_num
                                for y_kn1_idx = 1:y_num
                                    value_func_t = maxCostParam*ones(1,y_num);
                                    for y_k_idx = valid_y_idxs_t{x_k_idx,z_kn1_idx}
                                        P_h_vec_k_idx = adv_belief_transition_map_t(y_k_idx,y_kn1_idx,1,horizonIdx);
                                        value_func_t(y_k_idx) = C_HgHh_privacy_vecs_t(h_vec_k_idx,HhIdxs_given_P_h_vec_idx_t(P_h_vec_k_idx)) + ...
                                            exp_discounted_approxValueFn_in1_t(x_k_idx,z_kn1_idx,y_k_idx,h_vec_k_idx,horizonIdx,P_h_vec_k_idx);
                                    end
                                    
                                    [min_val,opt_y_idx] = min(value_func_t);
                                    valueFunction_i_t(x_k_idx,z_kn1_idx,y_kn1_idx,h_vec_k_idx,horizonIdx) = min_val;
                                    emu_strategy_i_t(x_k_idx,z_kn1_idx,y_kn1_idx,h_vec_k_idx,horizonIdx) = opt_y_idx;
                                end
                            end
                        end
                    end
                end
                
                valueFunction_i(:,:,:,:,:,P_h_vec_kn1_idx) = valueFunction_i_t;
                emu_strategy_i(:,:,:,:,:,P_h_vec_kn1_idx) = emu_strategy_i_t;
                send(parforProgress, incPercent);
            end
            max_val_inc = max(valueFunction_i(:)-valueFunction_in1(:));
            terminateParforProgress(max_val_inc);
            
            policy = struct;
            policy.emu_strategy = emu_strategy_i;
            policy.valueFunction = valueFunction_i;
            policy.valueFunction_in1 = valueFunction_in1;
            policy.max_val_inc = max_val_inc;
            policy.iter_idx = iter_idx;
            
            valueFunction_in1 = valueFunction_i;
            save(policy_fileName,'policy','params_t')
            if(max_val_inc<=params.min_max_val_inc || iter_idx ==max_valueFnIterations)
                fprintf('\t\tPerforming initialization step optimization : - ');
                valueFunction_in1 = policy.valueFunction_in1;
                emu_strategy_1 = zeros(x_num,z_num,h_vec_num,horizonsPerDay);
                
                discounted_approxValueFn_in1 = discountFactor*valueFunction_in1;
                exp_discounted_approxValueFn_in1 = zeros(x_num,z_num,y_num,h_vec_num,horizonsPerDay);
                initializeParforProgress();
                incPercent = (1/x_num)*50;
                parfor x_k_idx = 1:x_num
                    P_Zp1gZD_t = P_Zp1gZD;
                    valid_y_idxs_t = valid_y_idxs(x_k_idx,:);
                    possible_h_vec_idxs_t = possible_h_vec_idxs;
                    P_XkgXkn1HkHkn1_t = P_XkgXkn1HkHkn1;
                    P_Hk_vec_gHkn1_vec_t = P_Hk_vec_gHkn1_vec;
                    discounted_approxValueFn_in1_t = discounted_approxValueFn_in1;
                    P_H1_vec_idxs_t = P_H1_vec_idxs;
                    
                    exp_discounted_approxValueFn_in1_t = zeros(1,z_num,y_num,h_vec_num,horizonsPerDay);
                    for z_kn1_idx = 1:z_num
                        for y_k_idx = valid_y_idxs_t{z_kn1_idx}
                            d_k_idx = y_k_idx - x_k_idx - d_offset;
                            z_kp1_idx_prob = P_Zp1gZD_t(:,z_kn1_idx,d_k_idx);
                            for horizonIdx = 1:horizonsPerDay
                                P_h_vec_k_idx = P_H1_vec_idxs_t(y_k_idx,horizonIdx);
                                for h_vec_k_idx = 1:h_vec_num
                                    temp = 0;
                                    for h_vec_kp1_idx = possible_h_vec_idxs_t{h_vec_k_idx}
                                        temp = temp + P_Hk_vec_gHkn1_vec_t(h_vec_kp1_idx,h_vec_k_idx,horizonIdx)*...
                                            P_XkgXkn1HkHkn1_t(:,x_k_idx,h_vec_kp1_idx,h_vec_k_idx)'*discounted_approxValueFn_in1_t(:,:,y_k_idx,h_vec_kp1_idx,horizonIdx,P_h_vec_k_idx)*z_kp1_idx_prob;
                                    end
                                    exp_discounted_approxValueFn_in1_t(1,z_kn1_idx,y_k_idx,h_vec_k_idx,horizonIdx) = temp;
                                end
                            end
                        end
                    end
                    
                    exp_discounted_approxValueFn_in1(x_k_idx,:,:,:,:) = exp_discounted_approxValueFn_in1_t;
                    send(parforProgress, incPercent);
                end
                parfor x_k_idx = 1:x_num
                    C_HgHh_privacy_vecs_t = C_HgHh_privacy_vecs;
                    HhIdxs_given_P_h_vec_idx_t = HhIdxs_given_P_h_vec_idx;
                    P_H1_vec_idxs_t = P_H1_vec_idxs;
                    valid_y_idxs_t = valid_y_idxs(x_k_idx,:);
                    exp_discounted_approxValueFn_in1_t = exp_discounted_approxValueFn_in1(x_k_idx,:,:,:,:);
                    
                    emu_strategy_1_t = zeros(1,z_num,h_vec_num,horizonsPerDay);
                    for horizonIdx = 1:horizonsPerDay
                        for z_kn1_idx = 1:z_num
                            for h_vec_k_idx = 1:h_vec_num
                                value_func_t = maxCostParam*ones(1,y_num);
                                for y_k_idx = valid_y_idxs_t{z_kn1_idx}
                                    P_h_vec_k_idx = P_H1_vec_idxs_t(y_k_idx,horizonIdx);
                                    value_func_t(y_k_idx) = C_HgHh_privacy_vecs_t(h_vec_k_idx,HhIdxs_given_P_h_vec_idx_t(P_h_vec_k_idx)) + ...
                                        exp_discounted_approxValueFn_in1_t(1,z_kn1_idx,y_k_idx,h_vec_k_idx,horizonIdx);
                                end
                                
                                [~,opt_y_idx] = min(value_func_t);
                                emu_strategy_1_t(1,z_kn1_idx,h_vec_k_idx,horizonIdx) = opt_y_idx;
                            end
                        end
                    end
                    
                    emu_strategy_1(x_k_idx,:,:,:) = emu_strategy_1_t;
                    send(parforProgress, incPercent);
                end
                terminateParforProgress;
                policy.emu_strategy_1 = emu_strategy_1;
                save(policy_fileName,'policy','params_t')
                break;
            end
        end
        fprintf(['\t\tOptimization complete. Policy saved in: ',policy_fileName,'\n']);
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


