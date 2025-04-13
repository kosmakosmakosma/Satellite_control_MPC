
miu = 398600.4418;
a = 6778;
n = (miu/a^3)^0.5;
Ac = [[0, 0, 0, 1, 0, 0];
               [0, 0, 0, 0, 1, 0];
               [0, 0, 0, 0, 0, 1]; 
               [3*n^2, 0, 0, 0, 2*n, 0];
               [0, 0, 0, -2*n, 0, 0];
               [0, 0, -n^2, 0, 0, 0]];

Bc = [[0, 0, 0]; 
               [0, 0, 0]; 
               [0, 0, 0];                
               [1, 0, 0]; 
               [0, 1, 0];
               [0, 0, 1]];

Cc = [[1, 0, 0, 0, 0, 0];
               [0, 1, 0, 0, 0, 0];
               [0, 0, 1, 0, 0, 0]];



% Define the sampling time
Ts = 1; % Replace with your desired sampling time

% Transform the continuous system to a discrete system using the Tustin method
sys_c = ss(Ac, Bc, Cc, 0); % Continuous-time state-space system
% Compute the natural frequencies of the continuous system
eigenvalues = eig(Ac);
natural_frequencies = abs(imag(eigenvalues)); % Extract the imaginary parts
omega_n = max(natural_frequencies);

Ts = pi / (omega_n * 8); % Define the sampling time
disp(['Sampling time Ts: ', num2str(Ts)]);
Ts = 1;
disp(['Sampling time Ts: ', num2str(Ts)]);

sysd = c2d(sys_c, Ts, 'zoh');
[Ad, Bd, Cd, Dd] = ssdata(sysd); % Discrete-time state-space matrices

% Save the discrete-time state-space matrices Ad, Bd, Cd, and Dd as a .mat file
save('state_space_matrices.mat', 'Ad', 'Bd', 'Cd', 'Dd');

% Plot Bode plot of the continuous-time system
% figure;
% bode(sys_c, sysd);


% Load the state-space matrices from the .pkl file
% filename = 'state_space_matrices.pkl'; % Replace with the correct path to your .pkl file
% currentFolder = fileparts(mfilename('fullpath'));
% filename = fullfile(currentFolder, 'state_space_matrices.pkl');
% data = py.pickle.load(py.open(filename, 'rb'));

% Extract Ad, Bd, Cd, and Dd from the loaded data
% Ad = double(data{'Ad'});
% Bd = double(data{'Bd'});
% Cd = double(data{'Cd'});
% Dd = double(data{'Dd'});

% Display the matrices
% Define the desired pole locations
desiredPoles = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]; % Replace with your desired pole locations


% Compute the observer gain L using the place function
L = place(Ad', Cd', desiredPoles)';

% Display the observer gain matrix L
disp('Observer gain matrix L:');
disp(L);

% Compute the poles of the matrix (Ad - L*Cd)
observerPoles = eig(Ad - L * Cd);

% Display the poles
disp('Poles of (Ad - L*Cd):');
disp(observerPoles);

% Save the observer gain matrix L as a .mat file
save('observer_gain_L.mat', 'L');