function classMap = getCategorizedPaths(datasetPath)
    % Get a list of all subfolders (classes) in the dataset folder
    classList = dir(datasetPath);
    classList = classList([classList.isdir]); % Remove non-directory entries
    classList = classList(~ismember({classList.name}, {'.', '..'})); % Remove '.' and '..'

    % Initialize a containers.Map object to store class names and paths
    classMap = containers.Map;

    % Iterate through each class folder
    for i = 1:numel(classList)
        className = classList(i).name;
        classFolder = fullfile(datasetPath, className);

        % Get a list of all files in the class folder
        filePattern = fullfile(classFolder, '*.*');
        fileList = dir(filePattern);
        fileList = fileList(~[fileList.isdir]); % Remove subdirectories

        % Extract and store the paths of each file in the class
        filePaths = fullfile({fileList.folder}, {fileList.name});
        classMap(className) = filePaths;
    end
end
