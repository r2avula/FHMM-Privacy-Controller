function [fileName,fileExists,file_idx] = findFileName(params,fileNamePrefix,paramName,doIncrementalNaming)
fileExists = 0;
endReached = 0;
file_idx=1;
if(nargin==3)
    doIncrementalNaming = true;
end
if(doIncrementalNaming)
    while(~endReached)
        fileName = strcat(fileNamePrefix,num2str(file_idx,'%02d'),'.mat');
        if exist(fileName, 'file')
            file_idx=file_idx+1;
        else
            endReached = 1;
        end
    end
    endp1filename = fileName;
    while(~fileExists && file_idx>1)
        file_idx=file_idx-1;
        fileName = strcat(fileNamePrefix,num2str(file_idx,'%02d'),'.mat');
        %     variableInfo = who('-file', filename);
        %     if ismember(paramName, variableInfo)
        out = load(fileName,paramName);
        if(isfield(out,paramName) && isequaln(params,out.(paramName)))
            fileExists = 1;
        end
        %     end
    end
    if(~fileExists)
        fileName = endp1filename;
    end
else
    fileName = strcat(fileNamePrefix,num2str(file_idx,'%02d'),'.mat');
    if exist(fileName, 'file')
        out = load(fileName,paramName);
        if(isfield(out,paramName) && isequaln(params,out.(paramName)))
            fileExists = 1;
        end
    end
end
end