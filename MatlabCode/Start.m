% Path parameters
directory = 'Datasets/DatasetSystem/Test';
directory_save = 'Datasets/Crops/';

% For each image: process it, extract the crops and store it to
% directory_save.
files = ListFiles(directory);
for i = 1:length(files);
%for i = 13:13;    
    disp('---------')
    disp(i)
    im = imread(strcat(directory,'/',files(i).name));
    if ~exist(directory_save,'dir')
        mkdir(directory_save);
    end
    
    url_window_gt = strcat(directory, '/gt/gt.', files(i).name(1:size(files(i).name,2)-3), 'txt');
    
    disp('Processing file...')
    disp(files(i).name)
    disp(url_window_gt)
    
    %%
    annotations = [];
    Signs=[];
    fid = fopen(url_window_gt, 'r');
    tline = fgetl(fid);
    while ischar(tline)
        [A,c,e,ni]=sscanf(tline,'%f %f %f %f',4);
        sign_code = tline(ni+1:end);
        annotations = [annotations ; struct('x1', A(2), 'y1', A(1), 'x2', A(4), 'y2', A(3), 'sign', sign_code)];
        Signs=[Signs {sign_code}];
        tline = fgetl(fid);
    end
    fclose(fid);
    
    disp(annotations)
    
    %%
    pixelCandidates = ColorEnhancement(im);
    ExtractCrops(im, pixelCandidates, directory_save, files(i).name, annotations);
end




