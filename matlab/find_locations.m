function locs = find_locations(seq, m)
    locs = [];
    charged = 0;
    for i = 1:length(seq)
        if charged
            if seq(i) == checking_polarity
                count = count + 1;
                if count == m
                    locs = [locs i];
                    count = 1;
                    charged = 0;
                end
            else
                checking_polarity = seq(i);
                count = 1;
            end
        else
            checking_polarity = seq(i);
            charged = 1;
            count = 1;
        end
    end
end
