%% HW 4 Part 1
% Import and store the image files.
% Note the cropped images are 192x168
clc; clear all; close all

D = 'CroppedYale';
S = dir(fullfile(D));

% This matrix stores every image, from every sub-folder, in a large
% uint8 type matrix.
cropped_master_mat = [];

% Iterate through the contents of the folder, and one subfolder down.
% Please note that I obtained this code from Matlab's official forums as I
% do not have time to wade through the quirks of how matlab handles chars,
% workspace, and folders.

% The directory containing the subfolders
D = 'C:\Users\tyran\Dropbox\School Stuff\Winter 2020\AMATH_582\HW4\CroppedYale';

 %Makes a list of the directory of the folder containing the sub-folders.
S = dir(fullfile(D,'*'));

% removes the up/down dirs and makes a list of the folder names.
N = setdiff({S([S.isdir]).name},{'.','..'});

for i = 1:numel(N)
    
    T = dir(fullfile(D,N{i},'*')); %sub-folder directory
    C = {T(~[T.isdir]).name}; % files in subfolder.
    for j = 1:numel(C)
        F = fullfile(D,N{i},C{j});    
        % do whatever with file F.
        I = imread(F);
        I_col = I(:) - mean(I(:));
        cropped_master_mat = [cropped_master_mat I_col];

    end
end

%% Save the file

save('cropped_matrix.mat', 'cropped_master_mat')
writematrix(cropped_master_mat, 'cropped_matrix.csv')

%% Take an SVD and export
A = double(cropped_master_mat);
[U, S, V] = svd(A, 'econ');
save('centered_cropped_svd', 'U', 'S', 'V')


%% Uncropped face
% Import and arrange the uncropped faces and save the matrix file locally.
% Note these files are 243x320 pixels.
clc; clear all; close all;

D = 'C:\Users\tyran\Dropbox\School Stuff\Winter 2020\AMATH_582\HW4\yalefaces';
uncropped_master_mat = [];
T = dir(fullfile(D,'*')); %sub-folder directory
C = {T(~[T.isdir]).name}; % files in subfolder.
for j = 1:numel(C)
    F = fullfile(D,C{j});    
    % do whatever with file F.
    I = imread(F);
    I_col = I(:);
    uncropped_master_mat = [uncropped_master_mat I_col];

end


%% Save the matrix

save('uncropped_matrix.mat', 'uncropped_master_mat')
writematrix(uncropped_master_mat, 'uncropped_matrix.csv')

%% Take an svd and export

A = double(uncropped_master_mat);
[U, S, V] = svd(A, 'econ');
save('uncropped_svd', 'U', 'S', 'V')