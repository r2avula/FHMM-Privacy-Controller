function [points_out] = roundOff(points_in,precision)
belief_out_num = size(points_in,2);
a_num_t = size(points_in,1);
if(a_num_t>0)
    points_out = points_in/precision;
    for b_idx = 1:belief_out_num
        for a_idx_t = 1:a_num_t
            temp_t = floor(points_out(a_idx_t,b_idx));
            if mod(temp_t,2) == 0 && points_out(a_idx_t,b_idx) - temp_t == 0.5
                points_out(a_idx_t,b_idx) = round(points_out(a_idx_t,b_idx)/2)*2*precision;
            else
                points_out(a_idx_t,b_idx) = round(points_out(a_idx_t,b_idx))*precision;
            end
        end
    end
else
    points_out = points_in;
end
end