%%%%%%%%%%%%%%%%%%
%% This is the script to run CBP on the simulated dataset
%%%%%%%%%%%%%%%%%%

clear all;close all;
path = '../../../../CBP/spikesort_demo';
addpath(genpath(path))

% Load the parameters
numOfelements = 2;

params = load_default_parameters();
params.cbp.reweight_fn = @(x)x; % No reweighting
params.cbp.accuracy = 0.001;
params.cbp.num_reweights = 0;
params.cbp.greedy_p_value = -1;
params.cbp.lambda = 0.05;
params.cbp.compare_greedy = 0;
params.cbp.progress = 0;

% Initial waveforms
fs=100;

ts = [-5:10/fs:5];
init_waveforms={};
init_waveforms{1} = (ts.*exp(-ts.^2).*cos(2*pi*ts/4))';
init_waveforms{2} = (ts.*exp(-ts.^2))';



% Load the data
snippets={};
event_indices={};
snippet_centers=[];

elapsed_times={};

numOftrials = 1;
hit_errors=zeros(numOfelements, numOftrials);

T = 100;
for i=1:numOftrials
    idx=i
%     filename=['../../notebooks/data/csc_fs_',num2str(fs),'_T_',num2str(T),'_iter_',num2str(idx-1),'.mat'];
    filename=['../../notebooks/data/csc_demo.mat'];
    display(filename)
    info = load(filename);
    snippets{1}=info.data.data';
    snippet_centers=[round(length(info.data.data)/2)];
    event_indices{1}=info.data.indices;
    
    tic
    % Run CBP (There is going to be a single snippet)
    [spike_times, spike_amps, recon_snippets] = ...
        SpikesortCBP(snippets, snippet_centers, init_waveforms, params.cbp_outer, params.cbp);
    e=toc;
    
    amp_thresh = 1;
%     for j=1:numOfelements
%         spike_times{j} = (spike_times{j}-51)'/fs;
%         amps = spike_amps{j};
%         indices= find(amps>amp_thresh);
%         spike_times{j} = spike_times{j}(indices);
%         if length(event_indices{1}(j,:))== length(spike_times{j})
%             err = sum(abs(event_indices{1}(j,:) - spike_times{j}))
%             hit_errors(j,i) = err;
%         end
%     end
end



snippet_idx = 1;
figure()
hold on;
plot(snippets{snippet_idx},'r-','LineWidth',2)
plot(recon_snippets{snippet_idx},'b-','LineWidth',2)
hold off;
legend('Data','Reconstructed')
