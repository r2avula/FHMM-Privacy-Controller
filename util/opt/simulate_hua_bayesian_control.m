function [optimalControlData] = simulate_hua_bayesian_control(evalParams,bayesDetectorData,policy,smdata)
parforProgress = parallel.pool.DataQueue;
afterEach(parforProgress, @updateParforProgress);
percentDone = nan;
reverseStr = nan;
timeRemaining = nan;
proc_start = nan;
last_time_updated = nan;
msg1 = '';

params = evalParams.params;
storeAuxData = evalParams.storeAuxData;
storeBeliefs = evalParams.storeBeliefs;

minPowerDemandInW = params.minPowerDemandInW;
k_num = params.k_num;
x_num = params.x_num;
y_num = params.y_num;
h_vec_num = params.h_vec_num;
p_pu = params.p_pu;
x_offset = params.x_offset;
P_Zp1gZD = params.P_Zp1gZD;
d_offset = params.d_offset;
y_offset = params.x_offset;
P_Z0 = params.P_Z0;
paramsPrecision = params.paramsPrecision;
horizonsPerDay = params.horizonsPerDay;
minLikelihoodFilter = params.minLikelihoodFilter;
P_H1_vec_X1 = params.P_H1_vec_X1;

belief_transf = bayesDetectorData.belief_transf;
P_H1_vec_idxs = bayesDetectorData.P_H1_vec_idxs;
adv_belief_transition_map = bayesDetectorData.adv_belief_transition_map;

roundOffBelief_fn = @(x)roundOffBelief(x,paramsPrecision,[]);
exact_belief_k_default = roundOffBelief_fn(ones(h_vec_num,1)/h_vec_num);
P_H1_vec = zeros(h_vec_num,x_num,horizonsPerDay);
for horizonIdx = 1:horizonsPerDay    
    for x_idx = 1:x_num
        P_H1_vec_t = P_H1_vec_X1(:,x_idx,horizonIdx);
        P_H1_vec_sum_t = sum(P_H1_vec_t);
        if(P_H1_vec_sum_t>minLikelihoodFilter)
            P_H1_vec_t = roundOffBelief_fn(P_H1_vec_t/P_H1_vec_sum_t);            
        else
            P_H1_vec_t = exact_belief_k_default;
        end
        P_H1_vec(:,x_idx,horizonIdx) = P_H1_vec_t;
    end
end

numMCevalHorizons = evalParams.numMCevalHorizons;
k_num_idxs_in_horizons = params.k_num_idxs_in_horizons;

%% simulation
x_k_idxs = min(max(1,round((smdata-minPowerDemandInW)/p_pu)-x_offset),x_num);
modifiedSMdata = zeros(k_num,numMCevalHorizons);

if(storeAuxData)
    y_k_idxs = zeros(k_num,numMCevalHorizons);
    z_k_idxs = zeros(k_num,numMCevalHorizons);
    d_k_idxs = zeros(k_num,numMCevalHorizons);
end
if(storeBeliefs)
    belief_k_idxs = zeros(k_num,numMCevalHorizons);
end

Ykg_XkZkn1Ykn1HkPHhIdxkn1 = policy.emu_strategy;
Y1g_X1Z0H1 = policy.emu_strategy_1;

fprintf('\t\t\tSimulating controller : ');
initializeParforProgress;
incPercent = (1/numMCevalHorizons)*100;
parfor day_idx=1:numMCevalHorizons
    adv_belief_transition_map_t = adv_belief_transition_map;
    P_Zp1gZD_t = P_Zp1gZD;
    Ykg_XkZkn1Ykn1HkPHhIdxkn1_t = Ykg_XkZkn1Ykn1HkPHhIdxkn1;
    Y1g_X1Z0H1_t = Y1g_X1Z0H1;
    P_H1_vec_t = P_H1_vec;
    P_H1_vec_idxs_t = P_H1_vec_idxs;
    k_num_idxs_in_horizons_t = k_num_idxs_in_horizons;
    x_k_idxs_t = x_k_idxs(day_idx,:);
    modifiedSMdata_t = zeros(k_num,1);
    belief_transf_t = belief_transf;
    
    if(storeAuxData)
        y_k_idxs_t = zeros(k_num,1);
        z_k_idxs_t = zeros(k_num,1);
        d_k_idxs_t = zeros(k_num,1);
    else
        y_k_idxs_t = [];
        z_k_idxs_t = [];
        d_k_idxs_t = [];
    end
    if(storeBeliefs)
        belief_k_idxs_t = zeros(k_num,1);
    else
        belief_k_idxs_t = [];        
    end
    
    %     rng('shuffle')
    cumulative_distribution = cumsum(P_Z0);
    z_kn1_idx = find(cumulative_distribution>=rand(),1);    
    for horizonIdx = 1:horizonsPerDay
        k_idx_t = k_num_idxs_in_horizons_t(1,horizonIdx);
        x_k_idx_obs = x_k_idxs_t(k_idx_t);
        exact_belief_k = P_H1_vec_t(:,x_k_idx_obs,horizonIdx);
        temp_strat_t = reshape(Y1g_X1Z0H1_t(x_k_idx_obs,z_kn1_idx,:,horizonIdx),[],1);
        y_k_idx_star_prob = zeros(y_num,1);
        for y_idx = 1:y_num
            y_k_idx_star_prob(y_idx) = sum(exact_belief_k(temp_strat_t==y_idx));
        end
        cumulative_distribution = cumsum(y_k_idx_star_prob);
        y_k_idx_obs = find(cumulative_distribution>=rand(),1);
        adv_belief_k_idx = P_H1_vec_idxs_t(y_k_idx_obs,horizonIdx);
        
        d_k_idx_star = y_k_idx_obs - x_k_idx_obs - d_offset;
        z_k_idx_prob = P_Zp1gZD_t(:,z_kn1_idx,d_k_idx_star);
        cumulative_distribution = cumsum(z_k_idx_prob);
        z_k_idx = find(cumulative_distribution>=rand(),1);
        modifiedSMdata_t(k_idx_t) = minPowerDemandInW + (y_k_idx_obs + y_offset)*p_pu;
        
        if(storeAuxData)
            z_k_idxs_t(k_idx_t) = z_k_idx;
            d_k_idxs_t(k_idx_t) = d_k_idx_star;
            y_k_idxs_t(k_idx_t) = y_k_idx_obs;
        end
        if(storeBeliefs)
            belief_k_idxs_t(k_idx_t) = adv_belief_k_idx;
        end
        
        z_kn1_idx = z_k_idx;
        adv_belief_kn1_idx = adv_belief_k_idx;
        exact_belief_kn1 = exact_belief_k;
        y_kn1_idx_obs = y_k_idx_obs;
        x_kn1_idx_obs = x_k_idx_obs;        
        for k_idx = k_num_idxs_in_horizons_t(:,horizonIdx)'
            x_k_idx_obs = x_k_idxs_t(k_idx);
            exact_belief_k = belief_transf_t(:,:,x_k_idx_obs,x_kn1_idx_obs,horizonIdx)*exact_belief_kn1;
            sum_belief_k = sum(exact_belief_k);            
            if(sum_belief_k>minLikelihoodFilter)
                exact_belief_k = roundOffBelief_fn(exact_belief_k/sum_belief_k);
            else
                exact_belief_k = exact_belief_k_default;
            end                    
            temp_strat_t = reshape(Ykg_XkZkn1Ykn1HkPHhIdxkn1_t(x_k_idx_obs,z_kn1_idx,y_kn1_idx_obs,:,horizonIdx,adv_belief_kn1_idx),[],1);
            y_k_idx_star_prob = zeros(y_num,1);
            for y_idx = 1:y_num
                y_k_idx_star_prob(y_idx) = sum(exact_belief_k(temp_strat_t==y_idx));
            end
            cumulative_distribution = cumsum(y_k_idx_star_prob);
            y_k_idx_obs = find(cumulative_distribution>=rand(),1);
            adv_belief_k_idx = adv_belief_transition_map_t(y_k_idx_obs,y_kn1_idx_obs,adv_belief_kn1_idx,horizonIdx);
            
            d_k_idx_star = y_k_idx_obs - x_k_idx_obs - d_offset;
            z_k_idx_prob = P_Zp1gZD_t(:,z_kn1_idx,d_k_idx_star);
            cumulative_distribution = cumsum(z_k_idx_prob);
            z_k_idx = find(cumulative_distribution>=rand(),1);
            
            modifiedSMdata_t(k_idx) = minPowerDemandInW + (y_k_idx_obs + y_offset)*p_pu;
            if(storeAuxData)
                z_k_idxs_t(k_idx) = z_k_idx;
                d_k_idxs_t(k_idx) = d_k_idx_star;
                y_k_idxs_t(k_idx) = y_k_idx_obs;
            end
            if(storeBeliefs)
                belief_k_idxs_t(k_idx) = adv_belief_k_idx;
            end
            
            z_kn1_idx = z_k_idx;
            adv_belief_kn1_idx = adv_belief_k_idx;
            y_kn1_idx_obs = y_k_idx_obs;
            exact_belief_kn1 = exact_belief_k;
            x_kn1_idx_obs = x_k_idx_obs;
        end
    end
    
    modifiedSMdata(:,day_idx) = modifiedSMdata_t;
    if(storeAuxData)
        y_k_idxs(:,day_idx) = y_k_idxs_t;
        z_k_idxs(:,day_idx) = z_k_idxs_t;
        d_k_idxs(:,day_idx) = d_k_idxs_t;
    end
    if(storeBeliefs)
        belief_k_idxs(:,day_idx) = belief_k_idxs_t;
    end
    send(parforProgress, incPercent);
end
terminateParforProgress;

optimalControlData = struct;
optimalControlData.modifiedSMdata = modifiedSMdata';
if(storeAuxData)
    optimalControlData.d_k_idxs = d_k_idxs;
    optimalControlData.z_k_idxs = z_k_idxs;
    optimalControlData.y_k_idxs = y_k_idxs;
end
if(storeBeliefs)
    optimalControlData.belief_k_idxs = belief_k_idxs;
end

    function initializeParforProgress(~)
        percentDone = 0;
        timeRemaining = nan;
        reverseStr = 'Percent done = ';
        msg = sprintf('%3.2f', percentDone);
        msg1 = sprintf(', Est. time left = %ds',timeRemaining);
        msg = [msg,msg1];
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
        proc_start = tic;
        last_time_updated = -inf;
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

    function [proc_time] = terminateParforProgress()
        percentDone = 100;
        msg = sprintf('%3.2f', percentDone);
        proc_time = toc(proc_start);
        if(proc_time>86400)
            msg1 = sprintf('; Time taken = %3.1f days.        \n',proc_time/86400);
        elseif(proc_time>3600)
            msg1 = sprintf('; Time taken = %3.1f hours.      \n',proc_time/3600);
        elseif(proc_time>60)
            msg1 = sprintf('; Time taken = %3.1f minutes.  \n',(proc_time/60));
        else
            msg1 = sprintf('; Time taken = %3.1f seconds.  \n',(proc_time));
        end
        msg = [msg,msg1];
        fprintf([reverseStr, msg]);
    end
    
end

