addpath('C:\Users\maxim\Downloads\ott');
close all;
ott.warning('once');
ott.change_warnings('off');

n_medium = 1.33;        % Water
n_particle = 1.59;      % Polystyrene
wavelength0 = 532e-9;  % Vacuum wavelength
wavelength_medium = wavelength0 / n_medium;
radius = 1.0*wavelength_medium;

%beam = ott.BscPmGauss('NA', NA, 'polarisation', [ 1 1i ], ...
%     'index_medium', n_medium, 'wavelength0', wavelength0, 'power', 4.5e-3);


beam = ott.BscPlane(0,0, 'polarisation', [ 1 1i ], 'radius', 1e-3, ...
     'index_medium', n_medium, 'wavelength0', wavelength0, 'power', 4.5e-3);



beam.basis = 'regular';

figure();
subplot(1, 2, 1);
beam.visualise('axis', 'y');
subplot(1, 2, 2);
beam.visualise('axis', 'z');



beam.basis = 'incoming';

figure();
beam.visualiseFarfield('dir', 'neg');

T = ott.Tmatrix.simple('sphere', radius, 'wavelength0', wavelength0, ...
   'index_medium', n_medium, 'index_particle', n_particle);
sbeam = T * beam;

[force, torque] = ott.forcetorque(beam, sbeam);

nPc = 0.001 .* index_medium / 3e8;  % 0.001 W * n / vacuum_speed
force_SI = force .* nPc;