function [locations,X_idx] = subsample_threshold(x,y,t,p,m)
    ev = [x(:),y(:),t(:),p(:)];
    ev = sortrows(ev,3,"ascend");
    x = ev(:,1);
    y = ev(:,2);
    p = ev(:,4);
    x_max = max(x)+1;
    y_max = max(y)+1;
    X = cell(x_max,y_max);
    X_idx = cell(x_max,y_max);
    for ii = 1:length(x)
        X{x(ii)+1,y(ii)+1} = [X{x(ii)+1,y(ii)+1} p(ii)];
        X_idx{x(ii)+1,y(ii)+1} = [X_idx{x(ii)+1,y(ii)+1} ii];
    end
    locations = [];
    for ii_x = 1:x_max
        for ii_y = 1:y_max
            if(~isempty(X{ii_x,ii_y}))
                locs = find_locations(X{ii_x,ii_y}, m);
                locations = [locations X_idx{ii_x,ii_y}(locs)];
            end
        end
    end

end