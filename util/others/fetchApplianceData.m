function [appliancesData] = fetchApplianceData(config)
appliance_names = (config.applianceNames);
appliances_num = length(appliance_names);

private_appliance_names = (config.privateApplianceNames);
private_appliances_num = length(private_appliance_names);

applianceDataIDs = zeros(appliances_num,1);
private_applianceDataIDs = zeros(private_appliances_num,1);
for appliance_idx = 1:appliances_num
    appliance_name = cell2mat(appliance_names(appliance_idx));
    applianceID = getApplianceID(appliance_name);
    applianceDataIDs(appliance_idx) = applianceID;
end
for appliance_idx = 1:private_appliances_num
    appliance_name = cell2mat(private_appliance_names(appliance_idx));
    applianceID = getApplianceID(appliance_name);
    private_applianceDataIDs(appliance_idx) = applianceID;
end

[applianceDataIDs,sort_order] = sort(applianceDataIDs);
[~,private_appliance_idxs] = ismember(private_applianceDataIDs,applianceDataIDs);
appliance_names = appliance_names(sort_order);

dataset = config.dataset;
if(~isequal(dataset,'eco'))
    error('Not implemented!');
end

houseIndices = config.houseIndices;
if(iscell(houseIndices))
    houseIndices =  cell2mat(houseIndices);
    if(length(houseIndices)>1)
        error('Not implemented for more than one house!');
    end
end

dataStartDate = config.dataStartDate;
dataEndDate = config.dataEndDate;

house_str = num2str(houseIndices, '%02d');
start_date_num = datenum(dataStartDate, 'yyyy-mm-dd');
end_date_num = datenum(dataEndDate, 'yyyy-mm-dd');
dates_strs = datestr(start_date_num:end_date_num, 'yyyy-mm-dd');

availableDays = end_date_num-start_date_num+1;

slotIntervalInSeconds = config.slotIntervalInSeconds;
path_to_data = config.path_to_data;
evaluation_time = config.testEvaluationHourIndexBoundaries;
evaluation_time = (strsplit(cell2mat(evaluation_time)));
evalStartHourIndex = str2double(evaluation_time{1});
evalEndHourIndex = str2double(evaluation_time{2});
slotIntervalInHours = slotIntervalInSeconds/3600; %in hours

appliances_consumption = zeros(availableDays,24*3600/slotIntervalInSeconds,appliances_num);
appliancesRecordedDateMatrix = false(availableDays,appliances_num);

fileNamePrefix = 'cache/applianceData_';
dataParams = struct;
dataParams.dataset = dataset;
dataParams.startDate = dataStartDate;
dataParams.endDate = dataEndDate;
dataParams.evaluation_time = evaluation_time;
dataParams.house = houseIndices;
dataParams.slotIntervalInSeconds = slotIntervalInSeconds;
[filename,fileExists] = findFileName(dataParams,fileNamePrefix,'dataParams');
if(fileExists)
    load(filename,'cached_appliances_consumption','cached_appliance_IDs','cached_applianceRecordedDateMatrix');
else
    cached_appliances_consumption = [];    
    cached_appliance_IDs = [];
    cached_applianceRecordedDateMatrix = [];
end
cache_num = length(cached_appliance_IDs);

[cached_appliances,appliance_idxs_in_cache]  = ismember(applianceDataIDs,cached_appliance_IDs);
if(any(cached_appliances))    
    cached_appliance_idxs = find(cached_appliances);
    cached_appliances_num = length(cached_appliance_idxs);
    for appliance_idx_t = 1:cached_appliances_num
        appliance_idx = cached_appliance_idxs(appliance_idx_t);
        appliance_idx_in_cache = appliance_idxs_in_cache(appliance_idx);
        appliances_consumption(:,:,appliance_idx) = cached_appliances_consumption(:,:,appliance_idx_in_cache);
        appliancesRecordedDateMatrix(:,appliance_idx) = cached_applianceRecordedDateMatrix(:,appliance_idx_in_cache);
    end
end

if(any(~cached_appliances))
    not_cached_appliance_idxs = find(~cached_appliances);
    not_cached_appliances_num = length(not_cached_appliance_idxs);
    for appliance_idx_t = 1:not_cached_appliances_num
        appliance_idx = not_cached_appliance_idxs(appliance_idx_t);
        appliance_consumption = zeros(availableDays,24*3600/slotIntervalInSeconds);
        applianceID = applianceDataIDs(appliance_idx);
        
        num_of_days_selected = 0;
        idx_of_selected_days = zeros(1,availableDays);
        
        for day_idx = 1:availableDays
            plug_str = getPlugNr(applianceID, houseIndices, dataset);
            path_to_file = [path_to_data filesep dataset filesep 'plugs' filesep house_str filesep plug_str filesep dates_strs(day_idx,:) '.mat'];
            if exist(path_to_file, 'file')
                plug_consumption = load(path_to_file);
                temp_var_names = fieldnames(plug_consumption);
                plug_consumption = plug_consumption.(temp_var_names{1});
                plug_consumption = plug_consumption.('consumption');
                if (slotIntervalInSeconds > 1)
                    [mat_t,padded] = vec2mat(plug_consumption,slotIntervalInSeconds); %#ok<VEMAT>
                    assert(padded == 0, [num2str(slotIntervalInSeconds), ' is not a permissable interval (does not divide into 24h)']);
                    plug_consumption = mean(mat_t, 2);
                end
                plug_consumption = max(plug_consumption((evalStartHourIndex-1)/slotIntervalInHours+1:evalEndHourIndex/slotIntervalInHours),0);
                
                num_of_days_selected = num_of_days_selected + 1;
                idx_of_selected_days(num_of_days_selected) = day_idx;
                appliance_consumption(day_idx,:) = plug_consumption;
            end
        end
        idx_of_selected_days = idx_of_selected_days(1:num_of_days_selected);
                
        appliancesRecordedDateMatrix(idx_of_selected_days,appliance_idx) = true;
        appliances_consumption(:,:,appliance_idx) = appliance_consumption;
        cache_num = cache_num + 1;
        if(cache_num==1)
            cached_appliances_consumption = appliance_consumption;
            cached_applianceRecordedDateMatrix = appliancesRecordedDateMatrix(:,appliance_idx);
        else
            cached_appliances_consumption(:,:,cache_num) = appliance_consumption;        
            cached_applianceRecordedDateMatrix(:,cache_num) = appliancesRecordedDateMatrix(:,appliance_idx);
        end
        cached_appliance_IDs = [cached_appliance_IDs;applianceID]; %#ok<AGROW>
    end
    save(filename, 'cached_appliances_consumption','cached_appliance_IDs','cached_applianceRecordedDateMatrix','dataParams');
end

availableDateIdxs = find(sum(appliancesRecordedDateMatrix,2)==appliances_num);
dates_strs = datestr(availableDateIdxs + datenum(dataStartDate,'yyyy-mm-dd')-1 , 'yyyy-mm-dd');
appliances_consumption = appliances_consumption(availableDateIdxs,:,:);
total_consumption = sum(appliances_consumption,3);
availableDays = length(availableDateIdxs);

applianceON_powerThreshold = cell2mat((config.applianceON_powerThreshold));
applianceON_powerThreshold = applianceON_powerThreshold(sort_order);
appliances_state = zeros(availableDays,24*3600/slotIntervalInSeconds,appliances_num);
for appliance_idx = 1:appliances_num
    appliances_state(:,:,appliance_idx) = 1 + double(appliances_consumption(:,:,appliance_idx)>=applianceON_powerThreshold(appliance_idx));
end

appliancesData = struct;
appliancesData.appliance_names = appliance_names;
appliancesData.dates_strs = dates_strs;
appliancesData.total_consumption = total_consumption;
appliancesData.appliances_consumption = appliances_consumption;
appliancesData.appliances_state = appliances_state;
appliancesData.applianceDataIDs = applianceDataIDs;
appliancesData.applianceON_powerThreshold = applianceON_powerThreshold;
appliancesData.private_appliance_idxs = private_appliance_idxs;
end

