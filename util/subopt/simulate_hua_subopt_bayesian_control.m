function [optimalControlData] = simulate_hua_subopt_bayesian_control(evalParams,policy_afhmm,policy_dfhmm,...
    subopt_bayesDetectorData_afhmm,subopt_bayesDetectorData_dfhmm,smdata)
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
P_Hk_vec_gHkn1_vec = params.P_Hk_vec_gHkn1_vec;
minLikelihoodFilter = params.minLikelihoodFilter;
P_XgHvec = params.P_XgHvec;
P_H1_vec_X1 = params.P_H1_vec_X1;
Dx_num = params.Dx_num;
P_DXkgHkHkn1 = params.P_DXkgHkHkn1;
Dx_offset = params.Dx_offset;

belief_transf_a = zeros(h_vec_num,h_vec_num,x_num,horizonsPerDay);
for horizonIdx = 1:horizonsPerDay
    for y_k_idx = 1:y_num
        belief_transf_a(:,:,y_k_idx,horizonIdx) =  repmat(P_XgHvec(y_k_idx,:)',1,h_vec_num).*P_Hk_vec_gHkn1_vec(:,:,horizonIdx);
    end
end

belief_transf_d = zeros(h_vec_num,h_vec_num,Dx_num,horizonsPerDay);
for horizonIdx = 1:horizonsPerDay
    for dx_k_idx = 1:Dx_num
        belief_transf_d(:,:,dx_k_idx,horizonIdx) =  reshape(P_DXkgHkHkn1(dx_k_idx,:,:),h_vec_num,h_vec_num).*P_Hk_vec_gHkn1_vec(:,:,horizonIdx);
    end
end

roundOffBelief_fn = @(x)roundOffBelief(x,paramsPrecision,[]);
exact_belief_k_default = roundOffBelief_fn(ones(h_vec_num,1)/h_vec_num);
P_H1_vec = zeros(h_vec_num,x_num,horizonsPerDay);
for horizonIdx = 1:horizonsPerDay    
    for x_idx = 1:x_num
        P_H1_vec_t = P_H1_vec_X1(:,x_idx,horizonIdx);
        P_H1_vec_sum_t = sum(P_H1_vec_t);
        if(P_H1_vec_sum_t>minLikelihoodFilter)
            P_H1_vec(:,x_idx,horizonIdx) = roundOffBelief_fn(P_H1_vec_t/P_H1_vec_sum_t);
        else
            P_H1_vec(:,x_idx,horizonIdx) = exact_belief_k_default;
        end
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

Ykg_XkZkn1HkHhkn1_a = policy_afhmm.emu_strategy;
Y1g_X1Z0H1_a = policy_afhmm.emu_strategy_1;

Ykg_XkZkn1Ykn1HkHhkn1_d = policy_dfhmm.emu_strategy;
Y1g_X1Z0H1_d = policy_dfhmm.emu_strategy_1;

det_strategy_a = subopt_bayesDetectorData_afhmm.det_strategy;
det_strategy_1_a = subopt_bayesDetectorData_afhmm.det_strategy_1;
det_strategy_d = subopt_bayesDetectorData_dfhmm.det_strategy;
det_strategy_1_d = subopt_bayesDetectorData_dfhmm.det_strategy_1;

fprintf('\t\t\tSimulating controller : ');
initializeParforProgress;
incPercent = (1/numMCevalHorizons)*100;
parfor day_idx=1:numMCevalHorizons
    P_Zp1gZD_t = P_Zp1gZD;
    P_H1_vec_t = P_H1_vec;
    Ykg_XkZkn1HkHhkn1_a_t = Ykg_XkZkn1HkHhkn1_a;
    Y1g_X1Z0H1_a_t = Y1g_X1Z0H1_a;
    Ykg_XkZkn1Ykn1HkHhkn1_d_t = Ykg_XkZkn1Ykn1HkHhkn1_d;
    Y1g_X1Z0H1_d_t = Y1g_X1Z0H1_d;
    det_strategy_a_t = det_strategy_a;
    det_strategy_1_a_t = det_strategy_1_a;
    det_strategy_d_t = det_strategy_d;
    det_strategy_1_d_t = det_strategy_1_d;    
    belief_transf_a_t = belief_transf_a;
    belief_transf_d_t = belief_transf_d;
    
    k_num_idxs_in_horizons_t = k_num_idxs_in_horizons;
    x_k_idxs_t = x_k_idxs(day_idx,:);
    modifiedSMdata_t = zeros(k_num,1);
    
    if(storeAuxData)
        y_k_idxs_t = zeros(k_num,1);
        z_k_idxs_t = zeros(k_num,1);
        d_k_idxs_t = zeros(k_num,1);
    else
        y_k_idxs_t = [];
        z_k_idxs_t = [];
        d_k_idxs_t = [];
    end
    
    %     rng('shuffle')
    cumulative_distribution = cumsum(P_Z0);
    z_kn1_idx = find(cumulative_distribution>=rand(),1);    
    for horizonIdx = 1:horizonsPerDay
        k_idx_t = k_num_idxs_in_horizons_t(1,horizonIdx);
        x_k_idx_obs = x_k_idxs_t(k_idx_t);
        exact_belief_k = P_H1_vec_t(:,x_k_idx_obs,horizonIdx);
        temp_strat_a_t = reshape(Y1g_X1Z0H1_a_t(x_k_idx_obs,z_kn1_idx,:,horizonIdx),[],1);
        temp_strat_d_t = reshape(Y1g_X1Z0H1_d_t(x_k_idx_obs,z_kn1_idx,:,horizonIdx),[],1);
        y_k_idx_star_prob_a = zeros(y_num,1);
        y_k_idx_star_prob_d = zeros(y_num,1);
        for y_idx = 1:y_num
            y_k_idx_star_prob_a(y_idx) = sum(exact_belief_k(temp_strat_a_t==y_idx));
            y_k_idx_star_prob_d(y_idx) = sum(exact_belief_k(temp_strat_d_t==y_idx));
        end
        y_k_idx_star_prob = 0.5*y_k_idx_star_prob_a + 0.5*y_k_idx_star_prob_d;
        cumulative_distribution = cumsum(y_k_idx_star_prob);
        y_k_idx_obs = find(cumulative_distribution>=rand(),1);
        hh_k_idx_a = det_strategy_1_a_t(y_k_idx_obs,horizonIdx);
        hh_k_idx_d = det_strategy_1_d_t(y_k_idx_obs,horizonIdx);
                
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
        
        z_kn1_idx = z_k_idx;
        exact_belief_kn1 = exact_belief_k;
        y_kn1_idx_obs = y_k_idx_obs;
        hh_kn1_idx_a = hh_k_idx_a;
        hh_kn1_idx_d = hh_k_idx_d;   
        x_kn1_idx_obs = x_k_idx_obs;             
        for k_idx = k_num_idxs_in_horizons_t(:,horizonIdx)'
            x_k_idx_obs = x_k_idxs_t(k_idx);
            exact_belief_k_a = belief_transf_a_t(:,:,x_k_idx_obs,horizonIdx)*exact_belief_kn1;
            eta_a = sum(exact_belief_k_a);
            exact_belief_k_d = belief_transf_d_t(:,:,x_k_idx_obs-x_kn1_idx_obs-Dx_offset,horizonIdx)*exact_belief_kn1;
            eta_d = sum(exact_belief_k_d);
            
            if(eta_a>minLikelihoodFilter)
                exact_belief_k_a = roundOffBelief_fn(exact_belief_k_a/eta_a);
            else
                exact_belief_k_a = exact_belief_k_default;
            end
            if(eta_d>minLikelihoodFilter)
                exact_belief_k_d = roundOffBelief_fn(exact_belief_k_d/eta_d);
            else
                exact_belief_k_d = exact_belief_k_default;
            end
            
            eta_apd = (eta_a + eta_d);
            eta_a = eta_a/eta_apd;
            eta_d = eta_d/eta_apd;
            
            temp_strat_a_t = reshape(Ykg_XkZkn1HkHhkn1_a_t(x_k_idx_obs,z_kn1_idx,:,horizonIdx,hh_kn1_idx_a),[],1);
            temp_strat_d_t = reshape(Ykg_XkZkn1Ykn1HkHhkn1_d_t(x_k_idx_obs,z_kn1_idx,y_kn1_idx_obs,:,horizonIdx,hh_kn1_idx_d),[],1);
            y_k_idx_star_prob_a = zeros(y_num,1);
            y_k_idx_star_prob_d = zeros(y_num,1);
            for y_idx = 1:y_num
                y_k_idx_star_prob_a(y_idx) = sum(exact_belief_k_a(temp_strat_a_t==y_idx));
                y_k_idx_star_prob_d(y_idx) = sum(exact_belief_k_d(temp_strat_d_t==y_idx));
            end
            y_k_idx_star_prob = eta_a*y_k_idx_star_prob_a + eta_d*y_k_idx_star_prob_d;
            exact_belief_k = eta_a*exact_belief_k_a + eta_d*exact_belief_k_d;            
            cumulative_distribution = cumsum(y_k_idx_star_prob);
            y_k_idx_obs = find(cumulative_distribution>=rand(),1);
            
            hh_k_idx_a = det_strategy_a_t(y_k_idx_obs,hh_kn1_idx_a,horizonIdx);
            hh_k_idx_d = det_strategy_d_t(y_k_idx_obs-y_kn1_idx_obs-Dx_offset,hh_kn1_idx_d,horizonIdx);
            
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
            
            z_kn1_idx = z_k_idx;
            exact_belief_kn1 = exact_belief_k;
            y_kn1_idx_obs = y_k_idx_obs;
            hh_kn1_idx_a = hh_k_idx_a;
            hh_kn1_idx_d = hh_k_idx_d;
            x_kn1_idx_obs = x_k_idx_obs;
        end
    end
    
    modifiedSMdata(:,day_idx) = modifiedSMdata_t;
    if(storeAuxData)
        y_k_idxs(:,day_idx) = y_k_idxs_t;
        z_k_idxs(:,day_idx) = z_k_idxs_t;
        d_k_idxs(:,day_idx) = d_k_idxs_t;
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

