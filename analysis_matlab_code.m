%% HW 4 Part 1 Processing and Analysis
clc; clear all; close all

%% Cropped Faces

load('centered_cropped_svd.mat')

%% Sigma plots
sig = diag(S);
sig = sig(1:25);
% energy and log energy plots
figure(1)
subplot(1,2,1)
plot((sig / sum(sig)) * 100,'o', 'MarkerFaceColor', 'b')
xlabel('$\sigma_i$', 'Interpreter', 'latex')
ylabel('% of total energy')
set(gca, 'Fontsize', [10])
axis('square')
subplot(1,2,2)
semilogy((sig / sum(sig)) * 100,'o', 'MarkerFaceColor', 'b')
xlabel('$\sigma_i$', 'Interpreter', 'latex')
ylabel('Log of total energy')
set(gca, 'Fontsize', [10])
axis('square')
sgtitle('Percentage of total energy in each $\sigma$', 'interpreter', 'latex')

%% Cumulative energy
x_s = 1:25;
s_cum = cumsum(diag(sig) / sum(diag(sig)));


plot(x_s,s_cum, 'linewidth', 2)
xlabel('$i$ from $\sum_i\sigma_i$', 'Interpreter', 'latex')
ylabel('\% of energy by sum of $\sigma_i$', 'Interpreter', 'latex')
set(gca, 'Fontsize', [15])
xlim([1 25])
%% Plot the four dominant modes in grayscale

for k = 1:4
    subplot(2,2,k)
    im = mat2gray(-reshape(U(:,k), 192, 168) );
    imshow(im)
    title('POD Mode 1')
end
%%
cropped_average = mat2gray(-reshape(U(:,1), 192, 168) );
save('cropped_average.mat', 'cropped_average')
imshow(cropped_average)

%% Rank reduction comparison
close all
full_rank = U*S*V';
test_full = mat2gray(reshape( full_rank(:,27), 192, 168));
partial_rank = U(:,1:25)*S(1:25, 1:25)*V(:, 1:25)';
test_partial = mat2gray(reshape( partial_rank(:,27), 192, 168));
partial_rank2 = U(:,1:50)*S(1:50, 1:50)*V(:, 1:50)';
test_partial2 = mat2gray(reshape( partial_rank2(:,27), 192, 168));
partial_rank3 = U(:,1:100)*S(1:100, 1:100)*V(:, 1:100)';
test_partial3 = mat2gray(reshape( partial_rank3(:,27), 192, 168));


subplot(2,2,4)
imshow(test_full)
title('A full rank image')
subplot(2,2,1)
imshow(test_partial)
title('A rank 25 image')
subplot(2,2,2)
imshow(test_partial2)
title('A rank 50')
subplot(2,2,3)
imshow(test_partial3)
title('A rank 100')


%% Uncropped Faces
clc; clear all; close all;
load('uncropped_svd.mat')

%% Sigma plots
sig = diag(S);
sig = sig(1:25);
% energy and log energy plots
figure(1)
subplot(1,2,1)
plot((sig / sum(sig)) * 100,'o', 'MarkerFaceColor', 'b')
xlabel('$\sigma_i$', 'Interpreter', 'latex')
ylabel('% of total energy')
set(gca, 'Fontsize', [10])
axis('square')
subplot(1,2,2)
semilogy((sig / sum(sig)) * 100,'o', 'MarkerFaceColor', 'b')
xlabel('$\sigma_i$', 'Interpreter', 'latex')
ylabel('Log of total energy')
set(gca, 'Fontsize', [10])
axis('square')
sgtitle('Percentage of total energy in each $\sigma$', 'interpreter', 'latex')
%% Cumulative energy
x_s = 1:25;
s_cum = cumsum(diag(sig) / sum(diag(sig)));


plot(x_s,s_cum, 'linewidth', 2)
xlabel('$i$ from $\sum_i\sigma_i$', 'Interpreter', 'latex')
ylabel('\% of energy by sum of $\sigma_i$', 'Interpreter', 'latex')
set(gca, 'Fontsize', [15])
xlim([1 25])

%% Plot the four dominant modes in grayscale

for k = 1:4
    subplot(2,2,k)
    im = mat2gray(-reshape(U(:,k), 243, 320) );
    imshow(im)
    title('POD Mode 1')
end

%% Compare average face cropped and uncropped

subplot(1,2,2)
im = mat2gray(-reshape(U(:,1), 243, 320));
imshow(im)
title('Average Uncropped Face')
subplot(1,2,1)
imshow(cropped_average)
title('Average Cropped Face')
