function single_response()
    
    % Run spon activity to get non-zero neurones (only use if you want to
    % change randomness, otherwise comment out)
    % Seed 33 for amplitude 7 is about as close as i can get to og paper
    % for fig 2

    
    % seed = 30;
    % finding_spontaneous_activity_3(seed)
    
    %% SINGLE RESPONSE 3: Model - Version 3
    % To improve efficiency, relied on matrix operations for sensory input
    % and neurone sums
    % Removes a few 'for' loops
    % Rest of the model is unchanged

    %% MAIN PARAMETERS
        
    % N params
    NE = 100; % Number of excitatory neurons
    NI = 100; % Number of inhibitory neurons
    P = 15; % Number of columns
    
    % Tau params
    tau_E = 0.001; % Timescale of response for excitatory neurons
    tau_I = 0.001; % Timescale of response for inhibitory neurons
    tau_ref_E = 0.003; % Refractory period for excitatory neurons
    tau_ref_I = 0.003; % Refractory period for inhibitory neurons
    tau_rec = 0.8; % Recovery period for neurone resources
    U = 0.5; % Fraction of fraction of resources available
    
    % Synaptic efficacy params
    J0_II = -0.5;
    J0_EI = -4;
    J_IE = [0.5, 0.0035, 0.0015];
    J_EE = [6, 0.045, 0.015];


    %% Loading in: Non-zero spon excite activity and background input
    % Need an array to multiply by sensory input array to cancel out ones
    % that dont need input
    load('Data\model_input_data.mat', 'init_conds', 'zero_activity_indices', 'back_e', 'back_i')
    is_nonzero_spon = true(NE*P, 1);
    is_nonzero_spon(zero_activity_indices) = false;
    is_nonzero_spon = reshape(is_nonzero_spon, NE, P);
    
    
   
    %%  Sensory Input Params
    alpha = 2;
    delta_l = 5;
    lambda_c = 0.25;
    A = 4;   % Peak amplitude at column M 
    M = 8;   % Column 8 = M
    
    % Lambda s ( needs to be different for diff columns if non-symmetric)
    if A <= alpha
        lambda_s = lambda_c;
    else
        lambda_s = lambda_c + (A - alpha) / delta_l;
    end
       

    
    %% Initial conditions and Solver

    initial_conditions = init_conds;
    
    % Time span for the simulation
    tspan = [-1, 1]; 
    [t, y] = ode45(@(t, y) network_dynamics(t, y), tspan, initial_conditions);


    %% Plotting PS Response Amplitudes

    % matlab does row by column indexing
    % so rows are time points, we want columns relating to the right state variables

    % Convert time vector 't' from seconds to milliseconds
    t_ms = t * 1000;

    % Extract mean excitatory activity for each column 
    columns_to_plot = [6, 7, 8, 9, 10];
    all_excite_activities = y(:, 1:1500);

    column_8_excite = all_excite_activities(:, 701:800);
    mean_column_8 = mean(column_8_excite, 2);

    % Find the mean (or maximum) activity for the specified columns
    % mean_activities_columns = mean(excite_activity_columns, 2);  
    
    % Plot the mean activity for specified columns against time in milliseconds
    figure(1); 
    clf

    plot(t_ms, mean_column_8, 'linewidth',2 );
    xlim([-200, 400]); 
    ylim([0, 100]);
    title('Mean Excitatory Neuron Activity Over Time');
    xlabel('Time (Ms)' ,'FontSize', 20, 'FontWeight', 'bold')
    ylabel('Average E activity in column 8' ,'FontSize', 20, 'FontWeight', 'bold')
    title('Column 8 PS response to sensory tone input')
    set(gca, 'FontSize', 20, 'FontWeight', 'bold');
    set(gca, 'LineWidth', 2); 
    grid on;
    box off;
    set(gcf, 'Color', 'w'); 


    hold off; 
  
   
   
 
    
    %% ODE SOLVER FUNCTION

    function out = network_dynamics(t, y)
    
        % Unpack the state variables from y
        % Assuming y is organized with all E values first, then all I values,
        % followed by all X values, and finally all Y values for each of the P columns.
        E = reshape(y(1:NE*P), [NE, P]);
        I = reshape(y(NE*P+1:NE*P+NI*P), [NI, P]);
        X = reshape(y(NE*P+NI*P+1:NE*P+NI*P+NE*P), [NE, P]);
        Y = reshape(y(NE*P+NI*P+NE*P+1:end), [NI, P]);
    
        % Initialize derivatives
        dE = zeros(size(E));
        dI = zeros(size(I));
        dX = zeros(size(X));
        dY = zeros(size(Y));
    
    

        % Sensory input calc
        sensory_input = zeros(NE, P);
        if t >= 0 && t < 1
            for p = 1:P
                diff = abs(p - M);
                frac = diff / lambda_s;
                h = A * exp(-frac);
                sen_input = 1 * h;
                
                % Set this sensory input for all neurons in column p
                sensory_input(:, p) = sen_input;
            end
        end
        sen_input = sensory_input .* is_nonzero_spon;

        % Resource calcs
        X_prod = U .* X(:, :) .* E(:, :);
        Y_prod = U .* Y(:, :) .* I(:, :);

    
       % Calculate derivatives for each neuron and resource
    
        for p = 1:P

            EE_connections = zeros(NE, 5);
            IE_connections = zeros(NE, 5);

            % EE AND IE CONNECTIONS

            for R = -2:2
               
                column_index = p + R;
    
                % Skip the iteration if the column_index is less than 1 or greater than P
                if column_index < 1 || column_index > P
                    continue; 
                end

                EE_connections(:, R+3) = (J_EE(abs(R)+1)/NE) .* X_prod(:, p+R);
                IE_connections(:, R+3) = (J_IE(abs(R)+1)/NE) .* (E(:, p+R));

            end


            % BEGIN EXCITATORY NEURONE CALCS
            
            % EI Connections 
            EI_connections = ((J0_EI)./NI) .* (U .* Y(:, p) .* I(:, p));

            % Background input
            back_input_e = back_e(:, p);

            % Combine All
            total_input_e = back_input_e + sum(EE_connections(:)) + sum(EI_connections(:)) + sen_input(:, p);

            % Calculate Excitatory Neurone Derivatives
            dE(:, p) = (-E(:, p) + (1-(tau_ref_E.*E(:,p))).*relu(total_input_e))./ tau_E; % Activity
            dX(:, p) = ((1 - X(:, p)) ./ tau_rec) - X_prod(:, p); % Resources
      


            % BEGIN INHIBITORY NEURONE CALCS

            % II Connections
            II_connections =  ((J0_II)./NI) .* (I(:, p));
      
            % Combine All
            total_input_i = sum(IE_connections(:)) + sum(II_connections(:)) + back_i(:, p);


            % Inhibitory Neurone: Derivatives
            dI(:, p) = (-I(:, p) + (1-(tau_ref_I.*I(:,p))) .*relu(total_input_i)) ./ tau_I; % Activity
            dY(:, p) = ((1 - Y(:, p)) ./ tau_rec) - Y_prod(:, p); % Resources

        end
        
    
        % Flatten the derivative matrices back into a vector for ode45
        out = [dE(:); dI(:); dX(:); dY(:)];
    
    end
    
    
    

    %% NESTED FUNCTIONS

    function out = relu(z)
        out = max(z, 0);
    end
    
    
    
    
    function out = generate_uniform_values(n)
        out = -10 + (10 - (-10)) * rand(n, 1);
    end


end

