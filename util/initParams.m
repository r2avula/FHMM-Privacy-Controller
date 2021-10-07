function [params,viterbiParams,bayesDetectorParams] = initParams(config,appliancesData)
appliances_num = length(appliancesData.applianceDataIDs);
applianceStatesNum = 2*ones(appliances_num,1);

if(any(applianceStatesNum~=2))
    error('Not implemented!');    
end
ha_num = 2;
Dha_num = ha_num+1;
Dha_offset = -2;

h_vec_space = 1:applianceStatesNum(1);
for app_idx = 2:appliances_num
    temp_idxs = 1:applianceStatesNum(app_idx);
    h_vec_space = combvec(h_vec_space,temp_idxs);
end
h_vec_num = size(h_vec_space,2);

paramsPrecision = config.paramsPrecision;
p_pu = config.powerQuantPU; % in W

minPowerDemandInW = config.minPowerDemandInW; % in W
maxPowerDemandInW = min(config.maxPowerDemandInW, max(appliancesData.total_consumption(:)));
maxPowerDemandInW_compensated = maxPowerDemandInW - minPowerDemandInW; % in W
x_max_pu = floor(maxPowerDemandInW_compensated/p_pu);
x_min_pu = 0;
x_num = x_max_pu-x_min_pu+1;
x_offset = x_min_pu - 1;

y_num = x_num;
y_offset = x_offset;

Dx_range = -x_max_pu:x_max_pu;
Dx_num = length(Dx_range);
Dx_offset = -x_max_pu-1;

slotIntervalInSeconds = config.slotIntervalInSeconds;
slotIntervalInHours = slotIntervalInSeconds/3600; %in h
evalStartHourIndex = 1;
evalEndHourIndex = 24;
k_num = (evalEndHourIndex-evalStartHourIndex+1)/slotIntervalInHours; %Measurement slots
if(k_num~=floor(k_num))
    error('Wrong slotIntervalInSeconds setting!');
end

horizonsPerDay= config.horizonsPerDay;
k_num_in_horizon = k_num/horizonsPerDay;
k_num_idxs_in_horizons = zeros(k_num_in_horizon,horizonsPerDay);
temp = 0;
for horizonIdx = 1:horizonsPerDay
    k_num_idxs_in_horizons(:,horizonIdx) = temp + 1:temp +  k_num_in_horizon;
    temp = temp + k_num_in_horizon;
end

beliefSpacePrecision = config.beliefSpacePrecision;

% Prepare params
params = struct;
params.k_num = k_num;
params.horizonsPerDay = horizonsPerDay;
params.k_num_in_horizon = k_num_in_horizon;
params.k_num_idxs_in_horizons = k_num_idxs_in_horizons;
params.h_vec_num = h_vec_num;
params.x_num = x_num;
params.x_offset = x_offset;
params.p_pu = p_pu;
params.slotIntervalInHours = slotIntervalInHours;
params.paramsPrecision = paramsPrecision;
params.y_num = y_num;
params.y_offset = y_offset;
params.minPowerDemandInW = minPowerDemandInW;
params.h_vec_space = h_vec_space;
params.applianceStatesNum = applianceStatesNum;
params.appliances_num = appliances_num;
private_appliance_idxs = appliancesData.private_appliance_idxs;
params.private_appliance_idxs = private_appliance_idxs;

%% Get HMM params
params_t = params;
params_t.dataset = config.dataset;
params_t.dataset = config.dataset;
params_t.dataStartDate = config.dataStartDate;
params_t.dataEndDate = config.dataEndDate;
params_t.evaluation_time = (strsplit(cell2mat(config.testEvaluationHourIndexBoundaries)));
params_t.houseIndices = cell2mat(config.houseIndices);
params_t.applianceDataIDs = appliancesData.applianceDataIDs;
params_t.applianceON_powerThreshold = appliancesData.applianceON_powerThreshold;
fileNamePrefix = 'cache/hmmParams_';
[filename,fileExists] = findFileName(params_t,fileNamePrefix,'params_t');
if(fileExists)
    load(filename,'P_WgH','P_DWgDH','P_H0','P_HgHn1','P_XgHvec','possible_transitions_h_vec','P_XkgXkn1HkHkn1','P_Hk_vec_gHkn1_vec');
else    
    P_WgH = zeros(x_num,ha_num,appliances_num);
    P_DWgDH = zeros(Dx_num,Dha_num,appliances_num);
    P_H0 = zeros(ha_num,horizonsPerDay,appliances_num);
    P_HgHn1 = zeros(ha_num,ha_num,horizonsPerDay,appliances_num);
    
    appliances_consumption = appliancesData.appliances_consumption;
    total_consumption = appliancesData.total_consumption;
    appliances_state = appliancesData.appliances_state;
    
    for app_idx = 1:appliances_num
        el_data = appliances_consumption(:,:,app_idx)';
        state_data = appliances_state(:,:,app_idx)';
        x_k_idxs = min(max(1,round((el_data(:,:)-minPowerDemandInW)/p_pu)-x_offset),x_num);
        
        P_XgH_t = zeros(x_num,ha_num);
        for h_k_idx = 1:ha_num
            for x_k_idx = 1:x_num
                P_XgH_t(x_k_idx,h_k_idx) = sum(x_k_idxs(:)==x_k_idx & state_data(:)==h_k_idx);
            end
            
            temp_sum = sum(P_XgH_t(:,h_k_idx));
            if(temp_sum>0)
                P_XgH_t(:,h_k_idx) = P_XgH_t(:,h_k_idx)/temp_sum;
            else
                P_XgH_t(:,h_k_idx) = 1/x_num;
            end
        end
        P_WgH(:,:,app_idx) = P_XgH_t;
                
        diff_x_k = diff(x_k_idxs(:));
        diff_state = diff(state_data(:));
        
        P_DXgDH_t = zeros(Dx_num,Dha_num);
        for Dh_k_idx = 1:Dha_num
            Dh_k = Dh_k_idx + Dha_offset;
            for Dx_k_idx = 1:Dx_num
                Dx_k = Dx_k_idx + Dx_offset;
                P_DXgDH_t(Dx_k_idx,Dh_k_idx) = sum(diff_x_k==Dx_k & diff_state==Dh_k);
            end
            
            temp_sum = sum(P_DXgDH_t(:,Dh_k_idx));
            if(temp_sum>0)
                P_DXgDH_t(:,Dh_k_idx) = P_DXgDH_t(:,Dh_k_idx)/temp_sum;
            else
                P_DXgDH_t(:,Dh_k_idx) = 1/Dx_num;
            end
        end
        P_DWgDH(:,:,app_idx) = P_DXgDH_t;
                
        for horizonIdx = 1:horizonsPerDay
            if horizonIdx == 1
                k0_idx = k_num;
            else
                k0_idx = k_num_idxs_in_horizons(end,horizonIdx-1);
            end
                
            P_H0_t = zeros(ha_num,1);            
            for ha_idx = 1:ha_num
                P_H0_t(ha_idx) = sum(state_data(k0_idx,:)==ha_idx);
            end                               
            P_H0(:,horizonIdx,app_idx) = P_H0_t/sum(P_H0_t);
                        
            if horizonIdx == 1
                prev_h_state = [state_data(k_num,:);state_data(k_num_idxs_in_horizons(1:end-1,horizonIdx),:)];
            else
                prev_h_state = [state_data(k_num_idxs_in_horizons(end,horizonIdx-1),:);state_data(k_num_idxs_in_horizons(1:end-1,horizonIdx),:)];                
            end
            cur_h_state = state_data(k_num_idxs_in_horizons(:,horizonIdx),:);
            P_HgHn1_t = zeros(ha_num,ha_num);            
            for hkn1_idx = 1:ha_num
                for h_k_idx = 1:ha_num
                    P_HgHn1_t(h_k_idx,hkn1_idx) = sum(prev_h_state(:)==hkn1_idx & cur_h_state(:)==h_k_idx);
                end
                temp_sum = sum(P_HgHn1_t(:,hkn1_idx));
                if(temp_sum>0)
                    P_HgHn1_t(:,hkn1_idx) = P_HgHn1_t(:,hkn1_idx)/temp_sum;
                else
                    P_HgHn1_t(:,hkn1_idx) = 1/ha_num;
                end
            end
            P_HgHn1(:,:,horizonIdx,app_idx) = P_HgHn1_t;            
        end
    end
        
    P_XgHvec = zeros(x_num,h_vec_num);
    for h_vec_idx = 1:h_vec_num
        h_vec = h_vec_space(:,h_vec_idx);
        
        P_XgH_t = conv(P_WgH(:,h_vec(1),1), P_WgH(:,h_vec(2),2));
        P_XgH_t = P_XgH_t(1:x_num);
        P_XgH_t = P_XgH_t/sum(P_XgH_t);
        
        for app_idx = 3:appliances_num
            P_XgH_t = conv(P_XgH_t, P_WgH(:,h_vec(app_idx),app_idx));
            P_XgH_t = P_XgH_t(1:x_num);
            P_XgH_t = P_XgH_t/sum(P_XgH_t);
        end
        
        P_XgHvec(:,h_vec_idx) = roundOffBelief(P_XgH_t,paramsPrecision,[]);
    end
    
    availableDays = size(total_consumption,1);    
    x_k_idxs = min(max(1,round((total_consumption-minPowerDemandInW)/p_pu)-x_offset),x_num);
    state_data = zeros(availableDays,k_num);
    for dayIdx = 1:availableDays        
        [~,state_data(dayIdx,:)] = ismember(reshape(appliances_state(dayIdx,:,:),k_num,[]),h_vec_space','rows');
    end
    
    possible_transitions_h_vec = false(h_vec_num,h_vec_num);
    P_Hk_vec_gHkn1_vec = zeros(h_vec_num,h_vec_num,horizonsPerDay);    
    for h_vec_kn1_idx = 1:h_vec_num
        h_vec_kn1 = h_vec_space(:,h_vec_kn1_idx);
        P_Hk_vec_gHkn1_vec_t = zeros(h_vec_num,1,horizonsPerDay);
        for app_idx = 1:appliances_num
            h_vec_k = h_vec_kn1;
            h_vec_k(app_idx) = double(~logical(h_vec_k(app_idx) - 1)) + 1;
            [~, h_vec_k_idx] = ismember(h_vec_k',h_vec_space','row');
            
            possible_transitions_h_vec(h_vec_k_idx,h_vec_kn1_idx) = true;         
            for horizonIdx = 1:horizonsPerDay
                temp_t = P_HgHn1(h_vec_k(1),h_vec_kn1(1),horizonIdx,1)*P_HgHn1(h_vec_k(2),h_vec_kn1(2),horizonIdx,2);
                for app_idx_t = 3:appliances_num
                    temp_t = temp_t*P_HgHn1(h_vec_k(app_idx_t),h_vec_kn1(app_idx_t),horizonIdx,app_idx_t);
                end
                P_Hk_vec_gHkn1_vec_t(h_vec_k_idx,1,horizonIdx) = temp_t;
            end
        end
        
        h_vec_k = h_vec_kn1;
        h_vec_k_idx = h_vec_kn1_idx;
        possible_transitions_h_vec(h_vec_k_idx,h_vec_kn1_idx) = true;
        for horizonIdx = 1:horizonsPerDay
            temp_t = P_HgHn1(h_vec_k(1),h_vec_kn1(1),horizonIdx,1)*P_HgHn1(h_vec_k(2),h_vec_kn1(2),horizonIdx,2);
            for app_idx_t = 3:appliances_num
                temp_t = temp_t*P_HgHn1(h_vec_k(app_idx_t),h_vec_kn1(app_idx_t),horizonIdx,app_idx_t);
            end
            P_Hk_vec_gHkn1_vec_t(h_vec_k_idx,1,horizonIdx) = temp_t;
            
            
            temp_sum = sum(P_Hk_vec_gHkn1_vec_t(:,1,horizonIdx));
            if(temp_sum>0)
                P_Hk_vec_gHkn1_vec_t(:,1,horizonIdx) = P_Hk_vec_gHkn1_vec_t(:,1,horizonIdx)/temp_sum;
            else
                P_Hk_vec_gHkn1_vec_t(:,1,horizonIdx) = 1/h_vec_num;
            end            
        end
        P_Hk_vec_gHkn1_vec(:,h_vec_kn1_idx,:) = P_Hk_vec_gHkn1_vec_t;
    end
        
    x_k_idxs = x_k_idxs(:);
    state_data = state_data(:);
    x_kn1_idxs = x_k_idxs(1:end-1);
    x_k_idxs = x_k_idxs(2:end);
    h_kn1_idxs = state_data(1:end-1);
    h_k_idxs = state_data(2:end);
    
    P_XkgXkn1HkHkn1 = (1/x_num)*ones(x_num,x_num,h_vec_num,h_vec_num);
    for h_vec_kn1_idx = 1:h_vec_num
        h_vec_kn1 = h_vec_space(:,h_vec_kn1_idx);
        for app_idx = 1:appliances_num
            h_vec_k = h_vec_kn1;
            h_vec_k(app_idx) = double(~logical(h_vec_k(app_idx) - 1)) + 1;
            [~, h_vec_k_idx] = ismember(h_vec_k',h_vec_space','row');
            
            for xkn1_idx = 1:x_num
                P_XkXkn1_t = zeros(x_num,1);
                for x_k_idx = 1:x_num
                    P_XkXkn1_t(x_k_idx) = sum(x_kn1_idxs==xkn1_idx & x_k_idxs==x_k_idx & h_k_idxs == h_vec_k_idx & h_kn1_idxs == h_vec_kn1_idx);
                end
                temp_sum = sum(P_XkXkn1_t);
                if(temp_sum>0)
                    P_XkgXkn1HkHkn1(:,xkn1_idx,h_vec_k_idx,h_vec_kn1_idx) = P_XkXkn1_t/temp_sum;    
                end
            end
        end
        h_vec_k_idx = h_vec_kn1_idx;
        
        for xkn1_idx = 1:x_num
            P_XkXkn1_t = zeros(x_num,1);
            for x_k_idx = 1:x_num
                P_XkXkn1_t(x_k_idx) = sum(x_kn1_idxs==xkn1_idx & x_k_idxs==x_k_idx & h_k_idxs == h_vec_k_idx & h_kn1_idxs == h_vec_kn1_idx);
            end
            temp_sum = sum(P_XkXkn1_t);
            if(temp_sum>0)
                P_XkgXkn1HkHkn1(:,xkn1_idx,h_vec_k_idx,h_vec_kn1_idx) = P_XkXkn1_t/temp_sum;
            end
        end
    end
    
    save(filename,'P_WgH','P_DWgDH','P_H0','P_HgHn1','P_XgHvec','possible_transitions_h_vec','P_XkgXkn1HkHkn1','P_Hk_vec_gHkn1_vec','params_t');
end

C_HgHh = cell(appliances_num,1);
for app_idx = 1:appliances_num
    C_HgHh_t = eye(applianceStatesNum(app_idx));
    C_HgHh{app_idx} = C_HgHh_t;
end

C_HgHh_privacy = cell(appliances_num,1);
for app_idx = setdiff(1:appliances_num,private_appliance_idxs')
    C_HgHh_privacy{app_idx} = zeros(applianceStatesNum(app_idx));
end
for app_idx = private_appliance_idxs'
    C_HgHh_t = C_HgHh{app_idx};
    C_HgHh_privacy{app_idx} = C_HgHh_t;
end

minLogParam = config.minLogParam;
if(~isnumeric(minLogParam))
    minLogParam = str2double(minLogParam);
end

possible_h_vec_idxs = cell(h_vec_num,1);
for h_vec_idx = 1:h_vec_num
    possible_h_vec_idxs{h_vec_idx} = find(possible_transitions_h_vec(:,h_vec_idx))';
end

C_HgHh_privacy_vecs_app = zeros(h_vec_num,h_vec_num,appliances_num);
for h_vec_idx = 1:h_vec_num
    h_vec = h_vec_space(:,h_vec_idx);
    for hh_vec_idx = 1:h_vec_num
        hh_vec = h_vec_space(:,hh_vec_idx);
        for app_idx = 1:appliances_num
            C_HgHh_privacy_vecs_app(h_vec_idx,hh_vec_idx,app_idx) = C_HgHh_privacy{app_idx}(h_vec(app_idx),hh_vec(app_idx));
        end
    end
end
C_HgHh_privacy_vecs = sum(C_HgHh_privacy_vecs_app,3);


C_HgHh_vecs_app = zeros(h_vec_num,h_vec_num,appliances_num);
for h_vec_idx = 1:h_vec_num
    h_vec = h_vec_space(:,h_vec_idx);
    for hh_vec_idx = 1:h_vec_num
        hh_vec = h_vec_space(:,hh_vec_idx);
        for app_idx = 1:appliances_num
            C_HgHh_vecs_app(h_vec_idx,hh_vec_idx,app_idx) = C_HgHh{app_idx}(h_vec(app_idx),hh_vec(app_idx));
        end
    end
end
C_HgHh_vecs = sum(C_HgHh_vecs_app,3);

P_DXkgHkHkn1 = (1/Dx_num)*ones(Dx_num,h_vec_num,h_vec_num);
for h_vec_kn1_idx = 1:h_vec_num
    h_vec_kn1 = h_vec_space(:,h_vec_kn1_idx);
    for h_vec_k_idx = possible_h_vec_idxs{h_vec_kn1_idx}
        h_vec_k = h_vec_space(:,h_vec_k_idx);
        
        Dh_vec_idxs  = h_vec_k - h_vec_kn1 - Dha_offset;
        
        P_DX_t = conv(P_DWgDH(:,Dh_vec_idxs(1),1), P_DWgDH(:,Dh_vec_idxs(2),2),'same');
        P_DX_t = P_DX_t/sum(P_DX_t);        
        for app_idx = 3:appliances_num
            P_DX_t = conv(P_DX_t, P_DWgDH(:,Dh_vec_idxs(app_idx),app_idx),'same');
            P_DX_t = P_DX_t/sum(P_DX_t);
        end
        
        P_DXkgHkHkn1(:,h_vec_k_idx,h_vec_kn1_idx) = roundOffBelief(P_DX_t,paramsPrecision,[]);
    end
end

minLikelihoodFilter = config.minLikelihoodFilter;
P_H1_vec_X1 = zeros(h_vec_num,x_num,horizonsPerDay);
for horizonIdx = 1:horizonsPerDay
    P_H0_vec = zeros(h_vec_num,1);
    for h_vec_idx = 1:h_vec_num
        h_vec = h_vec_space(:,h_vec_idx);
        P_H0_vec(h_vec_idx) = P_H0(h_vec(1),horizonIdx,1)*P_H0(h_vec(2),horizonIdx,2);
        for app_idx = 3:appliances_num
            P_H0_vec(h_vec_idx) = P_H0_vec(h_vec_idx)*P_H0(h_vec(app_idx),horizonIdx,app_idx);
        end
    end
    
    for x_idx = 1:x_num
        for h_vec_idx = 1:h_vec_num
            P_H1_vec_X1(h_vec_idx,x_idx,horizonIdx) = P_XgHvec(x_idx,h_vec_idx)*P_Hk_vec_gHkn1_vec(h_vec_idx,:,horizonIdx)*P_H0_vec;
        end
    end
end

params.Dx_num = Dx_num;
params.Dha_offset = Dha_offset;
params.Dx_offset = Dx_offset;

params.P_DXkgHkHkn1 = P_DXkgHkHkn1;
params.P_H1_vec_X1 = P_H1_vec_X1;
params.beliefSpacePrecision = beliefSpacePrecision;
params.minLogParam = minLogParam;
params.C_HgHh = C_HgHh;
params.C_HgHh_privacy = C_HgHh_privacy;
params.C_HgHh_privacy_vecs = C_HgHh_privacy_vecs;
params.C_HgHh_privacy_vecs_app = C_HgHh_privacy_vecs_app;
params.C_HgHh_vecs_app = C_HgHh_vecs_app;
params.C_HgHh_vecs = C_HgHh_vecs;
params.P_H0 = P_H0;
params.P_HgHn1 = P_HgHn1;
params.P_WgH = P_WgH;
params.P_DWgDH = P_DWgDH;
params.P_XgHvec = P_XgHvec;
params.possible_h_vec_idxs = possible_h_vec_idxs;
params.P_XkgXkn1HkHkn1 = P_XkgXkn1HkHkn1;
params.P_Hk_vec_gHkn1_vec = P_Hk_vec_gHkn1_vec;
params.minLikelihoodFilter = minLikelihoodFilter;

maxCostParam = config.maxCostParam;
if(~isnumeric(maxCostParam))
    maxCostParam = str2double(maxCostParam);
end
params.maxCostParam = maxCostParam;

%% Get ESS params
e_pu = p_pu*slotIntervalInHours; % in Wh
batteryNominalVoltage = config.batteryNominalVoltage;
cell_nominalVoltage = config.cell_nominalVoltage; %in V - same for sim
cellsInSeries = ceil(batteryNominalVoltage/cell_nominalVoltage); % - same for sim
if(cellsInSeries~=floor(cellsInSeries))
    warning('Battery voltage is modified!');
end
batteryNominalVoltage = cellsInSeries*cell_nominalVoltage;% in V
converterEfficiency = (config.converterEfficiency)/100;  % - same for sim
batteryRatedCapacityInAh = config.batteryRatedCapacityInAh; %in Ah
cell_SOC_high = config.cell_SOC_high; % - same for sim
cell_SOC_low = config.cell_SOC_low; % - same for sim
cell_1C_capacityInAh = config.cell_1C_capacityInAh; %in Ah - same for sim
cell_1C_power = floor(cell_1C_capacityInAh*cell_nominalVoltage); %in W - same for sim
legsInParallel = round(batteryRatedCapacityInAh/cell_1C_capacityInAh);
d_max_ch_ess = cell_1C_power*legsInParallel*cellsInSeries/converterEfficiency;
d_max_disch_ess = -cell_1C_power*legsInParallel*cellsInSeries*converterEfficiency;

d_rated = maxPowerDemandInW_compensated;
d_max_ch = min(d_rated,d_max_ch_ess);
d_max_disch = max(-d_rated,d_max_disch_ess);

d_max_ch_pu = floor(d_max_ch/p_pu);
d_max_disch_pu = ceil(d_max_disch/p_pu);
out_pow_set = (d_max_disch_pu:d_max_ch_pu)*p_pu;
d_num = length(out_pow_set);
d_offset = d_max_disch_pu-1;
bat_pow_set = zeros(1,d_num);
for pow_idx = 1:d_num
    if(out_pow_set(pow_idx)<0)
        bat_pow_set(pow_idx) = out_pow_set(pow_idx)/converterEfficiency;
    else
        bat_pow_set(pow_idx) = out_pow_set(pow_idx)*converterEfficiency;
    end
end
z_cap = (batteryRatedCapacityInAh*batteryNominalVoltage); % in Wh
z_min_pu = floor(cell_SOC_low*z_cap/e_pu);
z_max_pu = floor(cell_SOC_high*z_cap/e_pu);
z_num = z_max_pu-z_min_pu+1;
z_offset = z_min_pu - 1;
soc_grid_boundaries = linspace(cell_SOC_low,cell_SOC_high,z_num+1);
soc_grid_bin_mean = zeros(z_num,1);
for bin_idx = 1:z_num
    soc_grid_bin_mean(bin_idx) = (soc_grid_boundaries(bin_idx) + soc_grid_boundaries(bin_idx+1))/2;
end

params.d_num = d_num;
params.d_offset = d_offset;
params.out_pow_set = out_pow_set;
params.bat_pow_set = bat_pow_set;
params.cellsInSeries = cellsInSeries;
params.legsInParallel = legsInParallel;
params.converterEfficiency = converterEfficiency;
params.batteryNominalVoltage = batteryNominalVoltage;
params.batteryRatedCapacityInAh = batteryRatedCapacityInAh;
params.z_num = z_num;
params.z_offset = z_offset;
params.e_pu = e_pu;
params.soc_grid_boundaries = soc_grid_boundaries;
params.soc_grid_bin_mean = soc_grid_bin_mean;
params.cell_1C_capacityInAh = cell_1C_capacityInAh;

meanCellInternalResistance = config.meanCellInternalResistance;
meanESSInternalResistance = meanCellInternalResistance*cellsInSeries/legsInParallel;
batterySelfDischargeRatePerMonth = 0.03; % factor in [0,1]
tau = 30*24/-log(1-batterySelfDischargeRatePerMonth); %h

ess_geom_spread_cdf_lim = config.ess_geom_spread_cdf_lim; % truncated geom dist
ess_geom_spread_range = config.ess_geom_spread_range; % only half-side including mode
ess_geom_spread_prob = 1 - (1-ess_geom_spread_cdf_lim)^(1/(ess_geom_spread_range-1+1));
paramsPrecision = config.paramsPrecision;
eSSnominalVoltage = cell_nominalVoltage*cellsInSeries;
energy_cap = cell_1C_capacityInAh*eSSnominalVoltage*legsInParallel;
P_Zp1gZD = zeros(z_num,z_num,d_num);
min_z_support_idxs =  zeros(z_num,d_num);
max_z_support_idxs =  zeros(z_num,d_num);
mean_ess_energyLossInWh_map = zeros(z_num,d_num);
for d_k_idx = 1:d_num
    for z_k_idx = 1:z_num
        bat_cur_pow = bat_pow_set(d_k_idx);
        ess_cur_pow_w_converter = out_pow_set(d_k_idx);
        cur_soc = soc_grid_bin_mean(z_k_idx);
        cur_soc_bin_idx = find(soc_grid_boundaries(2:end-1)>cur_soc,1);
        if(isempty(cur_soc_bin_idx))
            cur_soc_bin_idx =z_num;
        end
        if(cur_soc_bin_idx~=z_k_idx)
            error('something is wrong!');
        end
        
        if(bat_cur_pow == 0)
            gamma_simTime = (cur_soc-soc_grid_boundaries(1))/(cur_soc);
            simTimeInHours = min(-tau*log(1-gamma_simTime),slotIntervalInHours);
            gamma_simTime = 1 - exp(-simTimeInHours/tau);
            
            soc_kp1_estimate_3c = (1-gamma_simTime)*cur_soc;
            soc_kp1_estimate_3c = min(max(soc_kp1_estimate_3c,cell_SOC_low),cell_SOC_high);
            next_soc_bin_idx = find(soc_grid_boundaries(2:end-1)>soc_kp1_estimate_3c,1);
            if(isempty(next_soc_bin_idx))
                next_soc_bin_idx =z_num;
            end
            
            temp_prob_vec = zeros(z_num,1);
            temp_prob_vec(next_soc_bin_idx) = 1;
            P_Zp1gZD(:,z_k_idx,d_k_idx) = temp_prob_vec;
            
            
            z_support_idxs = find(temp_prob_vec>paramsPrecision);
            min_z_support_idx = z_support_idxs(1);
            max_z_support_idx = z_support_idxs(end);
            min_z_support_idxs(z_k_idx,d_k_idx) = min_z_support_idx;
            max_z_support_idxs(z_k_idx,d_k_idx) = max_z_support_idx;
            
            energyLossEstimate_3c = gamma_simTime*cur_soc*energy_cap +...
                (ess_cur_pow_w_converter*simTimeInHours);
            mean_ess_energyLossInWh_map(z_k_idx,d_k_idx) = energyLossEstimate_3c;
        elseif(bat_cur_pow>0)
            if(z_k_idx~= z_num)
                a_const = (tau*eSSnominalVoltage/2/meanESSInternalResistance)*...
                    (sqrt(max((eSSnominalVoltage*eSSnominalVoltage)+4*meanESSInternalResistance*bat_cur_pow,0))-eSSnominalVoltage);
                gamma_simTime = (soc_grid_boundaries(end) - cur_soc)*energy_cap/(a_const-cur_soc*energy_cap);
                simTimeInHours = min(-tau*log(1-gamma_simTime),slotIntervalInHours);
                gamma_simTime = 1 - exp(-simTimeInHours/tau);
                
                soc_kp1_estimate_3c = (1-gamma_simTime)*cur_soc + gamma_simTime*a_const/(energy_cap);
                soc_kp1_estimate_3c = min(max(soc_kp1_estimate_3c,cell_SOC_low),cell_SOC_high);
                next_soc_bin_idx = find(soc_grid_boundaries(2:end-1)>soc_kp1_estimate_3c,1);
                if(isempty(next_soc_bin_idx))
                    next_soc_bin_idx =z_num;
                end
                
                temp_prob_vec = zeros(z_num,1);
                temp_prob_vec(next_soc_bin_idx) = geopdf(0,ess_geom_spread_prob);
                for spread_idx = 1:ess_geom_spread_range-1
                    temp_prob = geopdf(spread_idx,ess_geom_spread_prob);
                    if(next_soc_bin_idx+spread_idx<=z_num)
                        temp_prob_vec(next_soc_bin_idx+spread_idx) = temp_prob;
                    end
                    if(next_soc_bin_idx-spread_idx>=1)
                        temp_prob_vec(next_soc_bin_idx-spread_idx) = temp_prob;
                    end
                end
                temp_prob_vec = roundOffBelief(temp_prob_vec/sum(temp_prob_vec),paramsPrecision,[]);
                
                P_Zp1gZD(:,z_k_idx,d_k_idx) = temp_prob_vec;
                energyLossEstimate_3c = gamma_simTime*cur_soc*energy_cap +...
                    (ess_cur_pow_w_converter*simTimeInHours - gamma_simTime*a_const);
                mean_ess_energyLossInWh_map(z_k_idx,d_k_idx) = energyLossEstimate_3c;
                
                z_support_idxs = find(temp_prob_vec>paramsPrecision);
                min_z_support_idx = z_support_idxs(1);
                max_z_support_idx = z_support_idxs(end);
                min_z_support_idxs(z_k_idx,d_k_idx) = min_z_support_idx;
                max_z_support_idxs(z_k_idx,d_k_idx) = max_z_support_idx;
            else
                P_Zp1gZD(:,z_k_idx,d_k_idx) = nan;
                mean_ess_energyLossInWh_map(z_k_idx,d_k_idx) = nan;
            end
        else
            if(z_k_idx~= 1)
                a_const = (tau*eSSnominalVoltage/2/meanESSInternalResistance)*...
                    (sqrt(max((eSSnominalVoltage*eSSnominalVoltage)+4*meanESSInternalResistance*bat_cur_pow,0))-eSSnominalVoltage);
                gamma_simTime = (soc_grid_boundaries(1) - cur_soc)*energy_cap/(a_const-cur_soc*energy_cap);
                simTimeInHours = min(-tau*log(1-gamma_simTime),slotIntervalInHours);
                gamma_simTime = 1 - exp(-simTimeInHours/tau);
                
                soc_kp1_estimate_3c = (1-gamma_simTime)*cur_soc + gamma_simTime*a_const/(energy_cap);
                soc_kp1_estimate_3c = min(max(soc_kp1_estimate_3c,cell_SOC_low),cell_SOC_high);
                next_soc_bin_idx = find(soc_grid_boundaries(2:end-1)>soc_kp1_estimate_3c,1);
                if(isempty(next_soc_bin_idx))
                    next_soc_bin_idx =z_num;
                end
                
                temp_prob_vec = zeros(z_num,1);
                temp_prob_vec(next_soc_bin_idx) = geopdf(0,ess_geom_spread_prob);
                for spread_idx = 1:ess_geom_spread_range-1
                    temp_prob = geopdf(spread_idx,ess_geom_spread_prob);
                    if(next_soc_bin_idx+spread_idx<=z_num)
                        temp_prob_vec(next_soc_bin_idx+spread_idx) = temp_prob;
                    end
                    if(next_soc_bin_idx-spread_idx>=1)
                        temp_prob_vec(next_soc_bin_idx-spread_idx) = temp_prob;
                    end
                end
                temp_prob_vec = roundOffBelief(temp_prob_vec/sum(temp_prob_vec),paramsPrecision,[]);
                
                energyLossEstimate_3c = gamma_simTime*cur_soc*energy_cap +...
                    (ess_cur_pow_w_converter*simTimeInHours - gamma_simTime*a_const);
                
                P_Zp1gZD(:,z_k_idx,d_k_idx) = temp_prob_vec;
                mean_ess_energyLossInWh_map(z_k_idx,d_k_idx) = energyLossEstimate_3c;
                z_support_idxs = find(temp_prob_vec>paramsPrecision);
                min_z_support_idx = z_support_idxs(1);
                max_z_support_idx = z_support_idxs(end);
                min_z_support_idxs(z_k_idx,d_k_idx) = min_z_support_idx;
                max_z_support_idxs(z_k_idx,d_k_idx) = max_z_support_idx;
            else
                P_Zp1gZD(:,z_k_idx,d_k_idx) = nan;
                mean_ess_energyLossInWh_map(z_k_idx,d_k_idx) = nan;
            end
        end
    end
end

P_Z0 = ones(z_num,1);
P_Z0 = P_Z0/sum(P_Z0);
P_Z0 = roundOffBelief(P_Z0/sum(P_Z0),paramsPrecision,[]);

valid_y_idxs = cell(x_num,z_num);
valid_z_idxs = cell(y_num,x_num,z_num);
for z_kn1_idx = 1:z_num
    for x_k_idx = 1:x_num
        valid_y_idxs_flag = true(y_num,1);
        for y_k_idx = 1:y_num
            d_k_idx = y_k_idx - x_k_idx - d_offset;
            if(d_k_idx<1 || d_k_idx >d_num)
                valid_y_idxs_flag(y_k_idx) = false;
            else
                z_kp1_idx_prob = P_Zp1gZD(:,z_kn1_idx,d_k_idx);
                if(any(isnan(z_kp1_idx_prob)) || sum(z_kp1_idx_prob)<paramsPrecision)
                    valid_y_idxs_flag(y_k_idx) = false;
                else
                    valid_z_idxs{y_k_idx,x_k_idx,z_kn1_idx} = find(z_kp1_idx_prob>=paramsPrecision)';
                end
            end
        end
        valid_y_idxs{x_k_idx,z_kn1_idx} = find(valid_y_idxs_flag)';
    end
end

params.P_Zp1gZD = P_Zp1gZD;
params.P_Z0 = P_Z0;
params.valid_y_idxs = valid_y_idxs;
params.valid_z_idxs = valid_z_idxs;

viterbiParams = struct;
viterbiParams.applianceDataIDs = sort(appliancesData.applianceDataIDs);
viterbiParams.h_vec_num = params.h_vec_num;
viterbiParams.k_num = params.k_num;
viterbiParams.horizonsPerDay = params.horizonsPerDay;
viterbiParams.k_num_idxs_in_horizons = params.k_num_idxs_in_horizons;
viterbiParams.minPowerDemandInW = params.minPowerDemandInW;
viterbiParams.P_Hk_vec_gHkn1_vec = params.P_Hk_vec_gHkn1_vec;
viterbiParams.minLogParam = params.minLogParam;
viterbiParams.x_num = params.x_num;
viterbiParams.p_pu = params.p_pu;
viterbiParams.x_offset = params.x_offset;
viterbiParams.P_XkgXkn1HkHkn1 = params.P_XkgXkn1HkHkn1;
viterbiParams.P_H1_vec_X1 = params.P_H1_vec_X1;

bayesDetectorParams = struct;
bayesDetectorParams.applianceDataIDs = sort(appliancesData.applianceDataIDs);
bayesDetectorParams.h_vec_num = params.h_vec_num;
bayesDetectorParams.k_num = params.k_num;
bayesDetectorParams.horizonsPerDay = params.horizonsPerDay;
bayesDetectorParams.k_num_idxs_in_horizons = params.k_num_idxs_in_horizons;
bayesDetectorParams.minPowerDemandInW = params.minPowerDemandInW;
bayesDetectorParams.P_Hk_vec_gHkn1_vec = params.P_Hk_vec_gHkn1_vec;
bayesDetectorParams.x_num = params.x_num;
bayesDetectorParams.p_pu = params.p_pu;
bayesDetectorParams.x_offset = params.x_offset;
bayesDetectorParams.P_XkgXkn1HkHkn1 = params.P_XkgXkn1HkHkn1;
bayesDetectorParams.paramsPrecision = params.paramsPrecision;
bayesDetectorParams.minLikelihoodFilter = params.minLikelihoodFilter;
bayesDetectorParams.P_H1_vec_X1 = params.P_H1_vec_X1;
end