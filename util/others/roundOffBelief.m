function [belief_out,belief_out_idxs] = roundOffBelief(belief_in,beliefSpacePrecision,dbs_data)
max_while = 10;
precision = 1e-6;
points_num = size(belief_in,2);
a_num = size(belief_in,1);
belief_in = belief_in./sum(belief_in);
belief_out = zeros(a_num,points_num);
belief_out_idxs = zeros(points_num,1);
if(~isempty(dbs_data))
    if(isstruct(dbs_data))
        dbs = dbs_data.dbs;
        nn_cells = dbs_data.nn_cells;
        use_facets = dbs_data.use_facets;
        if(use_facets)
            dbs_facets_full_dim = dbs_data.facets_full_dim;
            dbs_idx_of_facets = dbs_data.dbs_idx_of_facets;
        end
        for point_idx = 1:points_num            
            if(use_facets)
                tf = contains(dbs_facets_full_dim,belief_in(:,point_idx), true);
                constraints_satisfied = any(tf);
            else
                constraints_satisfied = false;
            end
            if(constraints_satisfied)
                belief_idx = dbs_idx_of_facets(tf);
                belief_out(:,point_idx) = dbs(:,belief_idx(1));
                belief_out_idxs(point_idx) = belief_idx(1);
            else
                tf = contains(nn_cells,belief_in(:,point_idx), true);
                if(any(tf))
                    belief_idx = find(tf);
                    belief_out(:,point_idx) = dbs(:,belief_idx);
                    belief_out_idxs(point_idx) = belief_idx;
                else
                    error('~constraints_satisfied')
                end
            end
        end
    else
        belief_space_T = dbs_data;
        for point_idx = 1:points_num
            rounded_belief_idx = knnsearch(belief_space_T,belief_in(:,point_idx)');
            belief_out(:,point_idx) = belief_space_T(rounded_belief_idx,:)';
            belief_out_idxs(point_idx) = rounded_belief_idx;
        end
    end
else
    beliefSpacePrecisionDigits = -log10(beliefSpacePrecision);
    if(beliefSpacePrecisionDigits == round(beliefSpacePrecisionDigits))
        for point_idx = 1:points_num
            belief_in_t = belief_in(:,point_idx);
            belief_in_tt = round(belief_in_t(1:end-1),beliefSpacePrecisionDigits);
            while_count = 0;
            while(sum(belief_in_tt)>1 && while_count<=max_while)
                belief_in_tt = belief_in_tt/sum(belief_in_tt);
                belief_in_tt = round(belief_in_tt,beliefSpacePrecisionDigits);
                while_count = while_count + 1;
            end
            if(sum(belief_in_tt)>1)
                nz_idxs = find(belief_in_tt>0)';
                residue = sum(belief_in_tt) - 1;
                for idx = nz_idxs(end:-1:1)
                    belief_in_ttt = max(0,belief_in_tt(idx) - residue);
                    residue = residue - (belief_in_tt(idx) - belief_in_ttt);
                    belief_in_tt(idx) = belief_in_ttt;
                    if(residue <=0)
                        break;
                    end
                end                
            end
            if(abs(1-sum(belief_in_tt))>precision)
                belief_out(:,point_idx) = [belief_in_tt;1-sum(belief_in_tt)];
            else
                belief_out(:,point_idx) = [belief_in_tt;0];
            end
        end
    else
        rounded_belief_in_floor = max(round(floor(belief_in/beliefSpacePrecision)*beliefSpacePrecision,9),0);
        rounded_belief_in_ceil = min(round(ceil(belief_in/beliefSpacePrecision)*beliefSpacePrecision,9),1);
        for point_idx = 1:points_num
            possible_rounded_beliefs = unique([rounded_belief_in_floor(1,point_idx),rounded_belief_in_ceil(1,point_idx)]);
            for a_idx_t = 2:a_num
                possible_rounded_beliefs_t = unique([rounded_belief_in_floor(a_idx_t,point_idx),rounded_belief_in_ceil(a_idx_t,point_idx)]);
                possible_rounded_beliefs = combvec(possible_rounded_beliefs,possible_rounded_beliefs_t);
            end
            
            possible_rounded_beliefs = possible_rounded_beliefs./sum(possible_rounded_beliefs);
            [possible_rounded_beliefs] = roundOff(possible_rounded_beliefs,beliefSpacePrecision);
            
            valid_rounded_beliefs_flag = sum(possible_rounded_beliefs,1)==1;
            
            if(any(valid_rounded_beliefs_flag))
                possible_rounded_beliefs = possible_rounded_beliefs(:,valid_rounded_beliefs_flag);
                rounded_belief_idx = knnsearch(possible_rounded_beliefs',belief_in(:,point_idx)');
                belief_out(:,point_idx) = possible_rounded_beliefs(:,rounded_belief_idx);
            else
                belief_in_t = belief_in(:,point_idx);
                rounded_belief_in_floor = max(round(floor(belief_in_t/beliefSpacePrecision)*beliefSpacePrecision,9)-beliefSpacePrecision,0);
                rounded_belief_in_ceil = min(round(ceil(belief_in_t/beliefSpacePrecision)*beliefSpacePrecision,9)+beliefSpacePrecision,1);
                possible_rounded_beliefs = unique([rounded_belief_in_floor(1),rounded_belief_in_ceil(1),possible_rounded_beliefs(1)]);
                for a_idx_t = 2:a_num
                    possible_rounded_beliefs_t = unique([rounded_belief_in_floor(a_idx_t),rounded_belief_in_ceil(a_idx_t),possible_rounded_beliefs(a_idx_t)]);
                    possible_rounded_beliefs = combvec(possible_rounded_beliefs,possible_rounded_beliefs_t);
                end
                
                possible_rounded_beliefs = possible_rounded_beliefs./sum(possible_rounded_beliefs);
                [possible_rounded_beliefs] = roundOff(possible_rounded_beliefs,beliefSpacePrecision);
                valid_rounded_beliefs_flag = sum(possible_rounded_beliefs,1)==1;
                
                if(any(valid_rounded_beliefs_flag))
                    possible_rounded_beliefs = possible_rounded_beliefs(:,valid_rounded_beliefs_flag);
                    rounded_belief_idx = knnsearch(possible_rounded_beliefs',belief_in(:,point_idx)');
                    belief_out(:,point_idx) = possible_rounded_beliefs(:,rounded_belief_idx);
                else
                    error('here');
                end
            end
        end
    end
end
end