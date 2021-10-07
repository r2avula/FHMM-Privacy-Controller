clear;
rng_id = 1;
simStartup(0,rng_id);
dbstop if error

storeAuxData = false;

opt_no_control_viterbi = 1;
subopt_bayesian_control_hyp_unaware = 1;
config_filename = '02_fridge_tv_lamp_freezer_stereo_600s_75w_30ah.yaml';

path_to_data = [pwd filesep 'data'];

%% parameter and data Initialization
config = ReadYaml(config_filename);
config.path_to_data = path_to_data;
appliancesData = fetchApplianceData(config);
[params,viterbiParams,bayesDetectorParams] = initParams(config,appliancesData);
appliances_num = params.appliances_num;
appliances_names_all = '';
for idx = 1:appliances_num-1
    appliances_names_all = [appliances_names_all,appliancesData.appliance_names{idx},', ']; %#ok<AGROW>
end
appliances_names_all = [appliances_names_all,appliancesData.appliance_names{appliances_num}];
private_appliance_idxs = params.private_appliance_idxs;
private_appliance_num = length(private_appliance_idxs);

testSMdata = appliancesData.total_consumption;
testGTdata = appliancesData.appliances_state;
availableEvalDays = size(testSMdata,1);

C_HgHh = params.C_HgHh;
C_HgHh_privacy = params.C_HgHh_privacy;
h_vec_space = params.h_vec_space;
k_num = params.k_num;
params_orig = params;

smdata = testSMdata;
gtdata = testGTdata;
numMCevalHorizons = availableEvalDays;

no_control_viterbi_detection_accuracy = zeros(1,appliances_num);
suboptimal_control_viterbi_detection_accuracy = zeros(1,appliances_num);

%% Viterbi MAP detection without controller
if(opt_no_control_viterbi)
    disp('Viterbi MAP detection without controller --- ');
    viterbiParams_t = viterbiParams;
    viterbiParams_t.numMCevalHorizons = numMCevalHorizons;
    viterbiParams_t.numMCevalHorizons = numMCevalHorizons;
    fileNamePrefix = 'cache/evalData_no_control_viterbi_';
    [fileName,fileExists] = findFileName(viterbiParams_t,fileNamePrefix,'viterbiParams_t');
    if(fileExists)
        fprintf(['\t\tEvaluation skipped. Data found in: ',fileName,'\n']);
        load(fileName,'det_accuracy_mean_viterbi','detected_sequence_viterbi');
    else              
        [detected_sequence_viterbi] = runViterbiDetection_controller_unaware(viterbiParams_t,smdata);
        
        detected_sequence_vec = reshape(detected_sequence_viterbi,[],1);
        det_accuracy_mean_viterbi =  zeros(1,appliances_num);
        for app_idx = 1:appliances_num
            C_HgHh_t = C_HgHh{app_idx};
            h_data_t = reshape(gtdata(:,:,app_idx),[],1);
            hh_data_t = h_vec_space(app_idx,detected_sequence_vec)';       
            ind = sub2ind(size(C_HgHh_t),h_data_t,hh_data_t);
            C_HgHh_t = C_HgHh_t(:);
            det_accuracy_mean_viterbi(app_idx) = mean(C_HgHh_t(ind));
        end
        
        save(fileName,'detected_sequence_viterbi','det_accuracy_mean_viterbi','viterbiParams_t')
        fprintf(['\t\tEvaluation complete. Data saved in: ',fileName,'\n']);
    end
    fprintf(['\tObserved detection accuracy of ',appliances_names_all,' : ',num2str(det_accuracy_mean_viterbi),'.\n']);
    fprintf(['\tMean observed detection accuracy : ',num2str(mean(det_accuracy_mean_viterbi)),'.\n']);
    no_control_viterbi_detection_accuracy(:) = det_accuracy_mean_viterbi;
    
    detected_sequence_vec = reshape(detected_sequence_viterbi,[],1);
    risk_mean_viterbi =  zeros(1,private_appliance_num);
    for p_idx = 1:private_appliance_num
        app_idx = private_appliance_idxs(p_idx);
        C_HgHh_privacy_t = C_HgHh_privacy{app_idx};
        h_data_t = reshape(gtdata(:,:,app_idx),[],1);
        hh_data_t = h_vec_space(app_idx,detected_sequence_vec)';
        ind = sub2ind(size(C_HgHh_privacy_t),h_data_t,hh_data_t);
        C_HgHh_privacy_t = C_HgHh_privacy_t(:);
        risk_mean_viterbi(p_idx) = mean(C_HgHh_privacy_t(ind));
    end
    fprintf(['\tMean observed risk : ',num2str(mean(risk_mean_viterbi)),'.\n']);
end

%% Bayesian suboptimal controller
numMCevalHorizons = 10000;
rng(rng_id,'twister');
eval_day_idxs = randi(availableEvalDays,1,numMCevalHorizons);
smdata = testSMdata(eval_day_idxs,:);
gtdata = testGTdata(eval_day_idxs,:,:);

if(subopt_bayesian_control_hyp_unaware)
    disp('SubOptimal Bayesian hyp. aware controller with discrete belief space --- ');    
    max_valueFnIterations = 20;
    params.discountFactor= 0.6;
    params.min_max_val_inc = 0.01;
    
    evalParams = struct;
    evalParams.params = params;
    evalParams.numMCevalHorizons = numMCevalHorizons;
    evalParams.storeAuxData = storeAuxData;
    fileNamePrefix = 'cache/evalData_hua_subopt_bayesian_control_';
    [fileName,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    if(fileExists)
        fprintf(['\t\tEvaluation skipped. Data found in: ',fileName,'\n']);
        load(fileName,'det_accuracy_mean_viterbi','detected_sequence_viterbi');
    else        
        [subopt_bayesDetectorData_afhmm] = get_subopt_bayes_detector_data_afhmm(params,max_valueFnIterations);
        [policy_afhmm] = get_subopt_hyp_aware_policy_afhmm(params,subopt_bayesDetectorData_afhmm,max_valueFnIterations);        
        [subopt_bayesDetectorData_dfhmm] = get_subopt_bayes_detector_data_dfhmm(params,max_valueFnIterations);      
        [policy_dfhmm] = get_subopt_hyp_aware_policy_dfhmm(params,subopt_bayesDetectorData_dfhmm,max_valueFnIterations); 
                
        [controlledData] = simulate_hua_subopt_bayesian_control(evalParams,policy_afhmm,policy_dfhmm,subopt_bayesDetectorData_afhmm,subopt_bayesDetectorData_dfhmm,smdata);
        viterbiParams_t = viterbiParams;
        viterbiParams_t.numMCevalHorizons = numMCevalHorizons;
        [detected_sequence_viterbi] = runViterbiDetection_controller_unaware(viterbiParams_t,controlledData.modifiedSMdata);        
        detected_sequence_vec_viterbi = reshape(detected_sequence_viterbi,[],1);
        det_accuracy_mean_viterbi =  zeros(1,appliances_num);
        for app_idx = 1:appliances_num
            C_HgHh_t = C_HgHh{app_idx};
            h_data_t = reshape(gtdata(:,:,app_idx),[],1);
            hh_data_t = h_vec_space(app_idx,detected_sequence_vec_viterbi)';
            ind = sub2ind(size(C_HgHh_t),h_data_t,hh_data_t);
            C_HgHh_t = C_HgHh_t(:);
            det_accuracy_mean_viterbi(app_idx) = mean(C_HgHh_t(ind));
        end
        
        save(fileName,'controlledData','detected_sequence_viterbi','det_accuracy_mean_viterbi','evalParams')
        fprintf(['\t\tEvaluation complete. Data saved in: ',fileName,'\n']);
    end
    fprintf(['\tObserved detection accuracy of viterbi - ',appliances_names_all,' : ',num2str(det_accuracy_mean_viterbi),'.\n']);
    fprintf(['\tMean observed detection accuracy viterbi : ',num2str(mean(det_accuracy_mean_viterbi)),'.\n']);
    
    suboptimal_control_viterbi_detection_accuracy(:) = det_accuracy_mean_viterbi;
    detected_sequence_vec = reshape(detected_sequence_viterbi,[],1);
    risk_mean_viterbi =  zeros(1,private_appliance_num);
    for p_idx = 1:private_appliance_num
        app_idx = private_appliance_idxs(p_idx);
        C_HgHh_privacy_t = C_HgHh_privacy{app_idx};
        h_data_t = reshape(gtdata(:,:,app_idx),[],1);
        hh_data_t = h_vec_space(app_idx,detected_sequence_vec)';
        ind = sub2ind(size(C_HgHh_privacy_t),h_data_t,hh_data_t);
        C_HgHh_privacy_t = C_HgHh_privacy_t(:);
        risk_mean_viterbi(p_idx) = mean(C_HgHh_privacy_t(ind));
    end
    fprintf(['\tMean observed risk viterbi : ',num2str(mean(risk_mean_viterbi)),'.\n']);    
end
