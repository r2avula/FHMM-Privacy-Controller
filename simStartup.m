function [] = simStartup(wait_time,rng_id)
if(nargin<2)
    wait_time = 0;
    rng_id = 1;
end
pathCell = regexp(path, pathsep, 'split');
test_dir = [pwd filesep 'util'];
onPath = any(strcmpi(test_dir, pathCell));
if (~onPath)        
    path(pathdef);
    addpath(genpath('util'));
    addpath(genpath('config'));    
end

if(wait_time>0)
    [~,p_pool] = evalc('gcp(''nocreate'');');
    if isempty(p_pool)
        fprintf('Parallel pool is missing. Type ''ENTER'' within %d seconds to start a new parallel pool ',wait_time);
        input_string = inputPrompter(wait_time);
        if(isempty(input_string))
            fprintf('Starting parallel pool...');
            [~,p_pool] = evalc('gcp;');
            poolsize = p_pool.NumWorkers;
            fprintf('Done. Connected to %d workers. \n',poolsize);
        else
            fprintf('Parallel pool not started.\n');
        end
    else
        poolsize = p_pool.NumWorkers;
        fprintf('Parallel pool is running. Connected to %d workers. \n',poolsize);
    end
end

rng(rng_id,'twister');
end
