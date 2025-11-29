% function [num, den]=LatLinMBlockSolver(m, l, bw, n, c, St)

function [num, den]=LatLinMBlockSolver(m, l, bw, n, c, St, tit)
    % m: mass
    % l: body length
    % w: body width
    % n: flapping frequency (Hz)
    % c: chord length (m)
    % St: both wings' total area (m^2)
    %% 
    
    g = 9.81; %acceleration due to gravity
 
    %forward flight speed (m/s) -- may need to correct for looming motion?
    u = 0;
    %vertical flight speed (m/s)
    w = 0;
    
    % phi: stroke amplitude (from average of control group from
    %      Fernandez/Hedrick '...wing asymmetry...')
    phi = 106*(pi/180);
    
    %rhoWing = massWing/areaSurfaceWing; % area mass density of wings
    % Radius of second moment of area of wing calculated 
    % using formula: sqrt(second moment of area of a wing / surface area of a wing)
    % where second moment of area = moment of inertia of a wing / wing mass density. 
    % Kim et al. 2015 reports this value as 24.63mm but using calculation reported
    % above, a value of 21mm was obtained. 
    %lengthSecondMomentArea = sqrt(((1/6*massWing*(lengthWing^2+lengthMeanChord^2))/rhoWing)/areaSurfaceWing);
    
    % r2: radius of second moment of wing area (radius of gyration --  sqrt[I about wing root/mass])
    r2 = 0.021; %(21mm)
    
    % Density of air at STP (kg/m^3)
    rho = 1.2754;
    
    %flapping period
    T = 1/n;
    
    % U:mean flapping velocity
    U = 2*phi*n*r2;
    
        
    %nondimensionalized mass
    mp = m/(rho*U*St*T);
    
    % non-dimensionalized acc due to grav
    gp = g*T/U;
    
    %% 
    
    % Ix, Iz, Ixz: normalized inertial tensor terms 
    % assuming ellipsoid (or cylinder?)
    Ixx = ((m/5)*(2*bw^2));
    Izz = ((m/5)*(l^2+bw^2));
    Ixz = 0;
    
%     % moment of inertia scaling to published data from Ellington 1984b
%     % allometric scaling factor 5/3 MOI/mass
%     scale = (m^(5/3)/(1.62^(5/3)));
%     Ixx = (2.55*10^-7)*scale;
%     Izz = (2.43*10^-7)*scale;
%     Ixz = (-3.34*10^-8)*scale;  


    % non-dimensionalizing MOI
    Ixxp = Ixx/(rho*U^2*St*c*T^2);
    Izzp =Izz/(rho*U^2*St*c*T^2);
    Ixzp = Ixz/(rho*U^2*St*c*T^2);
   
    % Inertial simplification for A matrix
    C1 = (Ixxp*Izzp-Ixzp^2);
    
    %% 
   
%     Non-dimensionalized aerodynamic derivatives: i.e. Yv is partial
%     derivative in Y (lateral force) to lateral velocity v
%     Xu_nonDim = -1.92; 
%     Xw_nonDim = -0.27;
%     Xq_nonDim = -0.13;
%     Zu_nonDim = -0.17;
%     Zw_nonDim = -2.13;
%     Zq_nonDim = -0.42;
%     Mu_nonDim = 0.87;
%     Mw_nonDim = 0.17;
%     Mq_nonDim = -0.23;
    Yv_nonDim = -1.00;
    Yp_nonDim = -0.17;
    Yr_nonDim = 0.44;
    Lv_nonDim = -0.11;
    Lp_nonDim = -1.00;
    Lr_nonDim = 0.30;
    Nv_nonDim = 0.20;
    Np_nonDim = 0.35;
    Nr_nonDim = -1.49;


    %Normalized, linearized, lateral equations of motion
    % states: x = [v (lateral velocity), p (yaw component of rotational velocity), 
    % r (roll component of rotational velocity), phi (yaw angle), y (lateral position)]
   
%     %probably incorrect 5 state system -- Izaak was right...maybe
%         A = [ Yv/mp Yp/mp Yr/mp-u gp 0;
%         (Iz*Lv+Ixz*Nv)/(Ix*Iz-Ixz^2) (Iz*Lp+Ixz*Np)/(Ix*Iz-Ixz^2) (Iz*Lr++Ixz*Nr)/(Ix*Iz-Ixz^2) 0 0;
%         (Ixz*Lv+Ix*Nv)/(Ix*Iz-Ixz^2) (Ixz*Lp+Ix*Np)/(Ix*Iz-Ixz^2) (Ixz*Lr+Ix*Nr)/(Ix*Iz-Ixz^2) 0 0;
%         0 1 0 0 0;
%         1 0 0 0 0];

%% 

    % 4 state system
  A = [Yv_nonDim/mp, (Yp_nonDim/mp)+w, (Yr_nonDim/mp)-u, gp;
      (Lv_nonDim*Izzp+Ixzp*Nv_nonDim)/C1, (Lp_nonDim*Izzp+Ixzp*Np_nonDim)/C1, (Izzp*Nr_nonDim+Nr_nonDim*Ixzp)/C1, 0;
      (Lv_nonDim*Izzp+Nv_nonDim*Ixzp)/C1, (Np_nonDim*Ixx+Lp_nonDim*Ixzp)/C1, (Nr_nonDim*Ixxp+Lr_nonDim*Ixzp)/C1, 0;
      0, 1, 0, 0];

    B = [1/m, 0, 0, 0;
        0, 1/((Ixx*Izz-Ixz^2)/Izz), Ixz/(Ixx*Izz-Ixz^2), 0;
        0, Ixz/(Ixx*Izz-Ixz^2), 1/((Ixx*Izz-Ixz^2)/Ixx), 0;
        0 0 0 0];
    
    % output matrix
    C = [1 0 0 0];
    
    D = [0, 0, 0, 0];
    
    %               NUM(s)          
    %       H(s) = -------- = C(sI-A)^(-1) B + D
    %               DEN(s)
    [num, den] = ss2tf(A,B,C,D,1);
    %% 
    
    %sys = ss(A,B,C,D);
    fvtool(num,den,'polezero');
    [z,p,k] = tf2zp(num,den);
    text(real(z)+.1,imag(z),'Zero')
    text(real(p)+.1,imag(p),'Pole')
    title([tit,' Mechanics Stability']);
    axis([-6 1 -3 3]);
    
end

% 
