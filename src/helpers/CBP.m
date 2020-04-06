%%%%%%%%%%%%%%%%%%
%% This is the script to run CBP on the simulated dataset
%%%%%%%%%%%%%%%%%%

clear all; close all;

path = '../../../../CBP/spikesort_demo';
addpath(genpath(path))

% Load the parameters
numOfelements = 1;

params = load_default_parameters();
params.cbp.reweight_fn = @(x)x; % No reweighting
params.cbp.accuracy = 0.01;
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
hit_errors=[];

T = 300;
events = [10, 20, 30, 40, 50];
for eidx=1:length(events)
    event = events(eidx);
    elapsed_times{eidx} = [];
    for i=1:25  
        display(i)
        filename=['../../notebooks/data/data_fs_',num2str(fs),'_T_',num2str(T),'_num_',num2str(event),'_iter_',num2str(i-1),'.mat'];
        info = load(filename);
        snippets{1}=info.data.data';
        snippet_centers=[length(info.data.data)/2];
        event_indices{1}=info.data.indices;

        tic
        % Run CBP (There is going to be a single snippet)
        [spike_times, spike_amps, recon_snippets] = ...
            SpikesortCBP(snippets, snippet_centers, init_waveforms, params.cbp_outer, params.cbp);
        e=toc;

        elapsed_times{eidx} = [elapsed_times{eidx} e];
    end
end

% T = [400,500];
% for tidx=1:length(T)
%     timestamp = T(tidx);
%     elapsed_times{tidx} = [];
%     for i=1:10   
%         display(i)
%         filename=['../../notebooks/data/data_fs_',num2str(fs),'_T_',num2str(timestamp),'_iter_',num2str(i-1),'.mat'];
%         info = load(filename);
%         snippets{1}=info.data.data';
%         snippet_centers=[length(info.data.data)/2];
%         event_indices{1}=info.data.indices;
% 
%         tic
%         % Run CBP (There is going to be a single snippet)
%         [spike_times, spike_amps, recon_snippets] = ...
%             SpikesortCBP(snippets, snippet_centers, init_waveforms, params.cbp_outer, params.cbp);
%         e=toc;
% 
%         elapsed_times{tidx} = [elapsed_times{tidx} e];
%     end
% end


% for i=1:numOfelements
%     % Recovering original timestamps
%     spike_times{i} = round(spike_times{i}*u_factor);
%     % Recovering original amplitudes
%     % The templates are not normalized => The amplitude has to be 'amplified'
% %     spike_amps{i} = spike_amps{i}*norm(init_waveforms{i});
% end

% figure();
% subplot(2,1,1);
% hold on;
% scatter(spike_times{1}, spike_amps{1},'x');
% scatter(event_indices(1,:), amp*ones(1,length(event_indices(1,:))),'o');
% xlim([0 datalen*u_factor])
% ylim([0 1])
% legend('CBP','Truth','Location','southeast');
% 
% subplot(2,1,2);
% hold on;
% scatter(spike_times{2}, spike_amps{2});
% scatter(event_indices(2,:), amp*ones(1,length(event_indices(2,:))),'o');
% xlim([0 datalen*u_factor])
% ylim([0 1])
% legend('CBP','Truth','Location','southeast');
% 

snippet_idx = 1;
figure()
hold on;
plot(snippets{snippet_idx},'r-','LineWidth',2)
plot(recon_snippets{snippet_idx},'b-','LineWidth',2)
hold off;
legend('Data','Reconstructed')
