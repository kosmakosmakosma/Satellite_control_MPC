
n = 0.01;
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

% Specify the path to the .pkl file
% Define the sampling time
Ts = 0.01; % Replace with your desired sampling time

% Transform the continuous system to a discrete system using the Tustin method
sys_c = ss(Ac, Bc, Cc, 0); % Continuous-time state-space system
sysd = c2d(sys_c, Ts);
[Ad, Bd, Cd, Dd] = ssdata(sysd); % Discrete-time state-space matrices

% Display the matrices
% Define the desired pole locations
desiredPoles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; % Replace with your desired pole locations


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