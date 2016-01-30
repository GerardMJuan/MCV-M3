function []=ExtractCrops(im, pixelCandidates, directory_save, name, gt_annotations)

%1. Connected components detection
%1.1 Obtain the connected components and their number of pixels.
[connected_components, num] = bwlabel(pixelCandidates, 8);
%Generate the bounding box
bb = regionprops(connected_components, 'BoundingBox');
Nwin = 0;
%5. For each bounding box...
candidates = [];
numberOfCandidates = 1;
windowCandidates = [];
for indexBoundingBoxes=1:size(bb)
    %5.1 Obtain the properties (for each bounding box)
    
    crop = imcrop(pixelCandidates, bb(indexBoundingBoxes).BoundingBox);
    
    %Obtain the BB values.
    x = bb(indexBoundingBoxes).BoundingBox(1);
    y = bb(indexBoundingBoxes).BoundingBox(2);
    temp_w = bb(indexBoundingBoxes).BoundingBox(3);
    temp_h = bb(indexBoundingBoxes).BoundingBox(4);
    temp_fillingRatio = nnz(crop)/(size(crop,1) * size(crop,2));
    win_size = 32;
    %Store the candidates
    %     if temp_fillingRatio >= 0.2901 && temp_fillingRatio <= 0.9866
    if temp_w >= 29.75 && temp_w <= 345.76
        if temp_h >= 29.46 && temp_h <= 253.39
            window = imcrop(pixelCandidates, [double(x) double(y) double(temp_w) double(temp_h)]);
            Nwin = Nwin + 1;
            fid = fopen( [directory_save,name(1:end-4),'_',num2str(Nwin),'.txt'],'w');
            candidates(numberOfCandidates) = indexBoundingBoxes;
            numberOfCandidates = numberOfCandidates + 1;
            windowCandidates = [windowCandidates; struct('x',double(x),'y',double(y),'w',double(temp_w),'h',double(temp_h))];
            max_size_crop = max(bb(indexBoundingBoxes).BoundingBox(3), bb(indexBoundingBoxes).BoundingBox(4));
            
            coords = [bb(indexBoundingBoxes).BoundingBox(1) - round(max_size_crop*4/win_size), ...
                bb(indexBoundingBoxes).BoundingBox(2) - round(max_size_crop*4/win_size),...
                max_size_crop + 2*round(max_size_crop*4/win_size), ...
                max_size_crop + 2*round(max_size_crop*4/win_size)];
            
            crop_im = imcrop(im, coords);
            crop_im = imresize(crop_im, [win_size win_size]);
            
            imwrite(crop_im,[directory_save,name(1:end-4),'_',num2str(Nwin),'.jpg']);
            
            window_coords = [struct('x1', bb(indexBoundingBoxes).BoundingBox(1), 'y1', bb(indexBoundingBoxes).BoundingBox(2), 'x2', bb(indexBoundingBoxes).BoundingBox(1)+bb(indexBoundingBoxes).BoundingBox(3), 'y2', bb(indexBoundingBoxes).BoundingBox(2)+bb(indexBoundingBoxes).BoundingBox(4))];
            

            %Now we want to look for the label
            %Check if there's a collision between window_coords and gt_annotations
            aux_ov = 0
            aux_sign = 0
            for j = 1:length(gt_annotations);
                overlapRatio = bboxOverlapRatio([gt_annotations(j).x1, gt_annotations(j).y1, abs(gt_annotations(j).x2 - gt_annotations(j).x1), abs(gt_annotations(j).y2 - gt_annotations(j).y1)]...
                ,[window_coords.x1, window_coords.y1, abs(window_coords.x2 - window_coords.x1), abs(window_coords.y2 - window_coords.y1)])
                
                mean_ov = mean(overlapRatio)
                
                % Which one has more overlap ratio?
                if (mean_ov > aux_ov)
                    aux_ov = mean_ov
                    aux_sign = j
                end
            end
            
            %disp(aux_ov)
            %disp(aux_sign)
            
            if (aux_sign > 0)
                y = gt_annotations(aux_sign).sign
            else
                y = 'Background'
            end
            
            %Store it to a text file
            fprintf( fid, '%f %f %f %f %s\r\n', bb(indexBoundingBoxes).BoundingBox(2), bb(indexBoundingBoxes).BoundingBox(1),...
                bb(indexBoundingBoxes).BoundingBox(2)+bb(indexBoundingBoxes).BoundingBox(4), bb(indexBoundingBoxes).BoundingBox(1)+bb(indexBoundingBoxes).BoundingBox(3), y);
            fclose(fid);
        end
    end
    %     end
end

end