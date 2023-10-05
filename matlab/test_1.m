
classMap = getCategorizedPaths(class_folder);

% Display class names and their corresponding paths
classNames = keys(classMap);
for i = 1:numel(classNames)
    className = classNames{i};
    fprintf('Class %s:\n', className);
    disp(classMap(className));
end
