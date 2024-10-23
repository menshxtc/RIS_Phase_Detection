%% 模拟环境选择
% switch Environment_choice
%     case 'Indoor (InH Office)'
%         Environment=1;
%     case 'Outdoor (UMi Street Canyon)'
%         Environment=2;
% end
Environment=1;

%% RIS悬挂位置选择
% switch Position_choice
%     case 'yz plane (Scenario 2)'
%         Scenario=2;
%     case 'xz plane (Scenario 1)'
%         Scenario=1;
% end
Scenario=1;

%% 信号频率选择
% switch Frequency_choice
%     case '28 GHz'
%         Frequency=28;
%     case '73 GHz'
%         Frequency=73;
% end
Frequency=28;
lambda=(3*10^8)/(Frequency*10^9);  % Wavelength;

%% RIS面板排列方式选择
% switch Array_choice
%     case 'Uniform Linear Array'
%         ArrayType=1;
%     case 'Uniform Planar Array'
%         ArrayType=2;
% end
ArrayType=1;

%% RIS元素数量
N=16;

%% 发射天线数量
Nt=4;

%% 接收天线数量
Nr=1;

%% 信道模拟次数
Nsym=10;

%% 发送者位置坐标
Tx_x=0;
Tx_y=25;
Tx_z=2;
Tx_xyz=[Tx_x,Tx_y,Tx_z];       % 单位：米

%% 接收者位置坐标
Rx_x=38;
Rx_y=48;
Rx_z=1;
Rx_xyz=[Rx_x,Rx_y,Rx_z];       % 单位：米

%% RIS位置坐标
RIS_x=40;
RIS_y=50;
RIS_z=2;
RIS_xyz=[RIS_x,RIS_y,RIS_z];   % 单位：米
d_T_RIS = norm(Tx_xyz-RIS_xyz);
d_RIS_R = norm(RIS_xyz-Rx_xyz);
d_T_R = norm(Tx_xyz-Rx_xyz);

cond = N*lambda/2; %far-field condition


%% 信道仿真主体

numIterations = 3;
H_N = complex(zeros(N, Nt, Nsym, numIterations));
G_N = complex(zeros(Nr, N, Nsym, numIterations));
D_N = complex(zeros(Nr, Nt, Nsym, numIterations));


% 使用 parfor 并行生成和存储矩阵
parfor repeat_n = 1:numIterations
    % 初始化临时三维矩阵
    H = complex(zeros(N, Nt, Nsym));
    G = complex(zeros(Nr, N, Nsym));
    D = complex(zeros(Nr, Nt, Nsym));

    for repeat = 1:Nsym
        % 调用 SimRIS_v18 生成三维复数矩阵 H, G1, D
        [H_temp, G1_temp, D_temp] = SimRIS_v18(Environment, Scenario, Frequency, ArrayType, N, Nt, Nr, Tx_xyz, Rx_xyz, RIS_xyz);
        G_temp = transpose(G1_temp);

        % 将临时矩阵的结果存储到 H, G, D 中
        H(:,:,repeat) = H_temp;
        G(:,:,repeat) = G_temp;
        D(:,:,repeat) = D_temp;
    end

    % 将结果存储到四维矩阵 H_N, G_N, D_N 中
    H_N(:,:,:,repeat_n) = H;
    G_N(:,:,:,repeat_n) = G;
    D_N(:,:,:,repeat_n) = D;
end

% 将结果分配到基础工作区中
assignin('base', 'H_N', H_N);
assignin('base', 'G_N', G_N);
assignin('base', 'D_N', D_N);





%% OFDM信号

% OFDM相关参数设置


% 信噪比SNR的计算



%% 核心信道生成函数
function [h,g,h_SISO]=SimRIS_v18(Environment,Scenario,Frequency,ArrayType,N,Nt,Nr,Tx_xyz,Rx_xyz,RIS_xyz)

% Inputs: Tx/Rx/RIS x-y-z Coordinates, Environment: Indoor/Outdoor (1/2), RIS Orientation (Scenario 1/2)
% Frequency (f),  Number of RIS Elements (N)

% Outputs: h, g, and h_SISO

% Environment=2;                     % 1 Indoor (InH - Indoor Office) / 2 Outdoor (UMi - Street Canyon)
% Scenario=1;                        % 1 (RIS in xz plane - left side wall) or 2 (RIS in yz plane - opposite wall)
% Frequency=28;                      % Operating Frequency (in GHz)
% N=64;                              % Number of RIS  Elements (sqrt(N) should be an integer)

lambda=(3*10^8)/(Frequency*10^9);  % Wavelength
k=2*pi/lambda;                     % Wavenumber
dis=lambda/2;                      % RIS element spacing (can be modified to observe its effect on h and g)
% Ge=pi;                           % Gain of RIS elements (maximum gain) - UPDATED VERSION 1.5

x_Tx=Tx_xyz(1);y_Tx=Tx_xyz(2);z_Tx=Tx_xyz(3);
x_Rx=Rx_xyz(1);y_Rx=Rx_xyz(2);z_Rx=Rx_xyz(3);
x_RIS=RIS_xyz(1);y_RIS=RIS_xyz(2);z_RIS=RIS_xyz(3);

if mod(sqrt(N),1)~=0
    error('N should be an integer power of 2')
end


if Environment==1 % INDOORS
    % Environment 1 (InH - Office) - NLOS
    n_NLOS=3.19;          % Path Loss Exponent (Indoor Office NLOS)
    sigma_NLOS=8.29;      % Shadow Fading Term (dB) (Indoor Office NLOS)
    b_NLOS=0.06;          % Path Loss Parameter (Indoor Office NLOS)
    f0=24.2;              % Path Loss Parameter (GHz) (Indoor Office NLOS)

    % Environment 1 (InH - Office) - LOS
    n_LOS=1.73;           % Path Loss Exponent (Indoor Office LOS)
    sigma_LOS=3.02;       % Shadow Fading Term (dB) (Indoor Office LOS)
    b_LOS=0;              % Path Loss Parameter (Indoor Office NLOS)

else  % OUTDOORS
    % Enviroment 2 (UMi - Street Canyon) - NLOS
    n_NLOS=3.19;          % Path Loss Exponent (UMi Street Canyon NLOS)
    sigma_NLOS=8.2;       % Shadow Fading Term (dB) (UMi Street Canyon NLOS)
    b_NLOS=0;             % Path Loss Parameter (UMi Street Canyon NLOS)
    f0=24.2;              % Useless since b_NLOS=0

    % Enviroment 2 (UMi - Street Canyon) - LOS
    n_LOS=1.98;          % Path Loss Exponent (UMi Street Canyon LOS)
    sigma_LOS=3.1;       % Shadow Fading Term (dB) (UMi Street Canyon LOS)
    b_LOS=0;             % Path Loss Parameter (UMi Street Canyon LOS)
end

% Parameter of Number of Clusters
if Frequency==28
    lambda_p=1.8;
elseif Frequency==73
    lambda_p=1.9;
end

% Element Radiation Pattern Parameters (Version 1.5)
q=0.285;
Gain=pi;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 1
% Calculate Tx-RIS LOS distance  and Generate LOS Component for h (Version 1.4)
d_T_RIS = norm(Tx_xyz-RIS_xyz);    % or sqrt(sum((RIS_xyz-Tx_xyz).^2))

% LOS Probability is Relatively Low for Indoors if d_T_RIS > 20
if Environment==1
    if z_RIS<z_Tx   % for ground level RIS
        % InH LOS Probability
        if d_T_RIS<= 1.2
            p_LOS=1;
        elseif 1.2<d_T_RIS && d_T_RIS<6.5
            p_LOS=exp(-(d_T_RIS-1.2)/4.7);
        else
            p_LOS=0.32*exp(-(d_T_RIS-6.5)/32.6);
        end

        I_LOS=randsrc(1,1,[1,0;p_LOS 1-p_LOS]);

    elseif z_RIS>=z_Tx % for an RIS mounted at a high place (100% LOS)
        I_LOS=1;
    end

elseif Environment==2
    % UMi LOS Probability
    p_LOS=min([20/d_T_RIS,1])*(1-exp(-d_T_RIS/39)) + exp(-d_T_RIS/39);

    I_LOS=randsrc(1,1,[1,0;p_LOS 1-p_LOS]);

end



if I_LOS==1
    %%%%%%%%%%%%

    % Calculate Tx Departure and RIS arrival angles to calculate array
    % response vectors
    if Scenario==1   % side-wall RIS

        % RIS arrival angles for LOS component
        I_phi=sign(x_RIS-x_Tx);
        phi_T_RIS_LOS = I_phi* atand ( abs( x_RIS-x_Tx) / abs(y_RIS-y_Tx) );

        I_theta=sign(z_Tx-z_RIS);
        theta_T_RIS_LOS=I_theta * asind ( abs (z_RIS-z_Tx )/ d_T_RIS );

        % Tx departure angles for LOS component
        I_phi_Tx=sign(y_Tx-y_RIS);
        phi_Tx_LOS = I_phi_Tx* atand ( abs( y_Tx-y_RIS) / abs(x_Tx-x_RIS) );

        I_theta_Tx=sign(z_Tx-z_RIS);
        theta_Tx_LOS=I_theta_Tx * asind ( abs (z_RIS-z_Tx )/ d_T_RIS );

    elseif Scenario==2 % opposite-wall RIS


        I_phi=sign(y_Tx-y_RIS);   % These are different from Scenario 1
        phi_T_RIS_LOS  = I_phi* atand ( abs(y_RIS-y_Tx ) / abs(  x_RIS-x_Tx ) );

        I_theta=sign(z_Tx-z_RIS);  % Same as Scenario 1
        theta_T_RIS_LOS=I_theta * asind ( abs (z_RIS-z_Tx )/ d_T_RIS );

        I_phi_Tx=sign(y_Tx-y_RIS);
        phi_Tx_LOS = I_phi_Tx* atand ( abs( y_RIS-y_Tx) / abs(x_RIS-x_Tx) );

        I_theta_Tx=sign(z_RIS-z_Tx);
        theta_Tx_LOS=I_theta_Tx * asind ( abs (z_RIS-z_Tx )/ d_T_RIS );

    end


    % Array Response Calculation (LOS) (Be Careful with sind/cosd)
    array_RIS_LOS=zeros(1,N);

    counter3=1;
    for x=0:sqrt(N)-1
        for y=0:sqrt(N)-1
            array_RIS_LOS(counter3)=exp(1i*k*dis*(x*sind(theta_T_RIS_LOS) + y*sind(phi_T_RIS_LOS)*cosd(theta_T_RIS_LOS) )) ;
            counter3=counter3+1;
        end
    end


    if ArrayType == 1
        array_Tx_LOS = zeros(1,Nt);
        counter3=1;
        for x=0:Nt-1
            array_Tx_LOS(counter3)=exp(1i*k*dis*(x*sind(phi_Tx_LOS)*cosd(theta_Tx_LOS) )) ;
            counter3=counter3+1;
        end
    elseif ArrayType == 2
        counter3=1;
        for x=0:sqrt(Nt)-1
            for y=0:sqrt(Nt)-1
                array_Tx_LOS(counter3)=exp(1i*k*dis*(x*sind(phi_Tx_LOS)*cosd(theta_Tx_LOS) + y*sind(theta_Tx_LOS))) ;
                counter3=counter3+1;
            end
        end

    end

    % Link Attentuation (LOS)

    % Note: This is different than FSPL (Shadowing/Waveguiding effect included with n < 2)
    L_dB_LOS=-20*log10(4*pi/lambda) - 10*n_LOS*(1+b_LOS*((Frequency-f0)/f0))*log10(d_T_RIS)- randn*sigma_LOS;

    L_LOS=10^(L_dB_LOS/10);

    % LOS COMPONENT GENERATED (with random phase) (practical PL and shadowing)
    % Element radiation pattern considered (Version 1.5)
    h_LOS=sqrt(L_LOS)*transpose(array_RIS_LOS)*array_Tx_LOS*exp(1i*rand*2*pi)*sqrt(Gain*(cosd(theta_T_RIS_LOS))^(2*q));


    %%%%%%%%%%%%
else
    h_LOS=0;
end

% Generated h_LOS can be used for both environments.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 2
% Generate Clusters/Sub-rays, Azimuth/Elevation Departure Angles and Cluster Distances

for generate=1:100  % To ensure that at least one scatterer exist

    % Number of Clusters - lambda_p was defined earlier: 1.8/1.9
    C=max([1,poissrnd(lambda_p)]);  % Poisson distributed

    % Number of Sub-rays per Cluster
    S=randi(30,1,C);
    % % for internal checks/tests (please comment)
    % C=1;
    % S=1;
    % Azimuth/Elevation Departure Angles
    phi_Tx=[ ];
    theta_Tx=[ ];
    phi_av=zeros(1,C);
    theta_av=zeros(1,C);
    for counter=1:C
        phi_av(counter)  = rand*180-90;     % mean azimuth
        theta_av(counter)= rand*90-45;      % 均值仰角  (needs update & most of the clusters are above ceiling)
        % for outdoors this might not be a big concern
        % cluster angles: First S(1) belongs to Cluster 1, Next S(2) belongs to Cluster 2....
        phi_Tx   = [phi_Tx,log(rand(1,S(counter))./rand(1,S(counter)))*sqrt(25/2) + phi_av(counter)];
        theta_Tx = [theta_Tx,log(rand(1,S(counter))./rand(1,S(counter)))*sqrt(25/2) + theta_av(counter)];
    end
    % Cluster Distances
    a_c=1+rand(1,C)*(d_T_RIS-1);    % Cluster distances uniform [1,d_T_RIS] % can be modified later
    % Correction on Cluster Locations for Indoors
    if Environment ==1
        % Room dimensions (Indoor Hotspot)
        dim=[75,50,3.5];                  % x-y dimensions recommended by 5G Channel Model, height is assumed as 3.5 m
        % Reducing Distances for Outside Clusters - Check Cluster Coordinates with Mean Angles
        Coordinates=zeros(C,3);          % for Clusters
        Coordinates2=zeros(sum(S),3);    % for Scatterers
        for counter=1:C
            loop=1;
            Coordinates(counter,:)=[x_Tx + a_c(counter)*cosd(theta_av(counter))*cosd(phi_av(counter)),...
                y_Tx - a_c(counter)*cosd(theta_av(counter))*sind(phi_av(counter)),...
                z_Tx + a_c(counter)*sind(theta_av(counter))] ;
            while Coordinates(counter,3)>dim(3) || Coordinates(counter,3)<0 ||  Coordinates(counter,2)>dim(2) ||  Coordinates(counter,2)<0  ||  Coordinates(counter,1)>dim(1) ||  Coordinates(counter,1)<0
                a_c(counter)=    0.8*a_c(counter)  ;     % reduce distance 10%-20% if coordinates are not met!
                % Note: While the above 10% reduction ensures that all clusters are in the range, to ensure that most of the scatterers are
                % in the range, we may consider 20% or 30% reduction. Scatter correction saves this issue.
                Coordinates(counter,:)=[x_Tx + a_c(counter)*cosd(theta_av(counter))*cosd(phi_av(counter)),...
                    y_Tx - a_c(counter)*cosd(theta_av(counter))*sind(phi_av(counter)),...
                    z_Tx + a_c(counter)*sind(theta_av(counter))] ;
                %    loop=loop+1
            end
            %     % Plot Clusters ( comment figures in mass generation of channels )
            % plot3(Coordinates(counter,1),Coordinates(counter,2),Coordinates(counter,3),'mdiamond')
            % hold on;
        end
    elseif Environment==2 % Outdoors
        % Reducing Distances for Outside Clusters - Check Cluster Coordinates with Mean Angles
        Coordinates=zeros(C,3);          % for Clusters
        Coordinates2=zeros(sum(S),3);    % for Scatterers
        for counter=1:C
            loop=1;
            Coordinates(counter,:)=[x_Tx + a_c(counter)*cosd(theta_av(counter))*cosd(phi_av(counter)),...
                y_Tx - a_c(counter)*cosd(theta_av(counter))*sind(phi_av(counter)),...
                z_Tx + a_c(counter)*sind(theta_av(counter))] ;
            while  Coordinates(counter,3)<0    % only underground clusters will be ignored
                a_c(counter)=    0.8*a_c(counter)  ;     % reduce distance 10%-20% if coordinates are not met!
                % Note: While the above 10% reduction ensures that all clusters are in the range, to ensure that most of the scatterers are
                % in the range, we may consider 20% or 30% reduction.
                Coordinates(counter,:)=[x_Tx + a_c(counter)*cosd(theta_av(counter))*cosd(phi_av(counter)),...
                    y_Tx - a_c(counter)*cosd(theta_av(counter))*sind(phi_av(counter)),...
                    z_Tx + a_c(counter)*sind(theta_av(counter))] ;
                %    loop=loop+1
            end
            %     % Plot Clusters (comment figures in mass generation of channels (with repeat))
            % plot3(Coordinates(counter,1),Coordinates(counter,2),Coordinates(counter,3),'mdiamond')
            % hold on;
        end
    end
    % Plot Scatterers
    a_c_rep=[];
    for counter3=1:C
        a_c_rep=[a_c_rep,repmat(a_c(counter3),1,S(counter3))];
    end
    for counter2=1:sum(S)
        Coordinates2(counter2,:)=[x_Tx + a_c_rep(counter2)*cosd(theta_Tx(counter2))*cosd(phi_Tx(counter2)),...
            y_Tx - a_c_rep(counter2)*cosd(theta_Tx(counter2))*sind(phi_Tx(counter2)),...
            z_Tx + a_c_rep(counter2)*sind(theta_Tx(counter2))] ;

        %       % comment figures in mass generation of channels
        %     plot3(Coordinates2(counter2,1),Coordinates2(counter2,2),Coordinates2(counter2,3),'m.')
        %     hold on;
    end



    % Correction on Scatters
    % You may ignore the scatterers outside the walls for Enviroment 1 (Indoors) or underground for Enviroment 2 (Outdoors)

    if Environment==1
        ignore=[];

        for counter2=1:sum(S)
            if Coordinates2(counter2,3)>dim(3) || Coordinates2(counter2,3)<0 ||  Coordinates2(counter2,2)>dim(2) ||  Coordinates2(counter2,2)<0  ||  Coordinates2(counter2,1)>dim(1) ||  Coordinates2(counter2,1)<0
                ignore=[ignore,counter2];   % contains the indices of scatterers that can be ignored
            end
        end

        % updated indices
        indices=setdiff(1:sum(S),ignore);    % the set of active scatterer indices
        M_new=length(indices);               % number of IOs inside the room or above ground



    elseif Environment==2

        ignore=[];

        for counter2=1:sum(S)
            if  Coordinates2(counter2,3)<0   % only underground scatterers
                ignore=[ignore,counter2];   % contains the indices of scatterers that can be ignored
            end
        end

        % updated indices
        indices=setdiff(1:sum(S),ignore);    % the set of active scatterer indices
        M_new=length(indices);               % number of IOs inside the room or above ground

    end

    % if you want to revert back Version 1.1 from Version 1.2, simply set
    % indices=1:sum(S);
    % M_new=sum(S);

    % Neccessary Loop to have at least one scatterer
    if M_new>0 % if M_new==0 --> all scatters are outside
        break  % break generate=1:100 % if M_new >0 we are OK (at least one scatter)
    end

end  % for generate=1:100
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 3 (Common for Enviroments 1 and 2)
% Calculate Arrival Angles for the RIS and the Link Distances

phi_cs_RIS=zeros(1,sum(S));
theta_cs_RIS=zeros(1,sum(S));
phi_Tx_cs=zeros(1,sum(S));
theta_Tx_cs=zeros(1,sum(S));
b_cs=zeros(1,sum(S));
d_cs=zeros(1,sum(S));

if Scenario==1   % side-wall RIS

    for counter2=indices

        b_cs(counter2)=norm(RIS_xyz-Coordinates2(counter2,:));   % Distance between Scatterer and RIS
        d_cs(counter2)=a_c_rep(counter2)+b_cs(counter2);         % Total distance Tx-Scatterer-RIS

        I_phi=sign(x_RIS-Coordinates2(counter2,1));
        phi_cs_RIS(counter2)  = I_phi* atand ( abs( x_RIS-Coordinates2(counter2,1)) / abs(y_RIS-Coordinates2(counter2,2)) );

        I_theta=sign(Coordinates2(counter2,3)-z_RIS);
        theta_cs_RIS(counter2)=I_theta * asind ( abs (z_RIS-Coordinates2(counter2,3) )/ b_cs(counter2) );

        I_phi_Tx_cs=sign(y_Tx-Coordinates2(counter2,2));
        phi_Tx_cs(counter2) = I_phi_Tx_cs* atand ( abs( Coordinates2(counter2,2)-y_Tx) / abs(Coordinates2(counter2,1)-x_Tx) );

        I_theta_Tx_cs=sign(Coordinates2(counter2,3)-z_Tx);
        theta_Tx_cs(counter2)=I_theta_Tx_cs * asind ( abs (Coordinates2(counter2,3)-z_Tx )/ a_c_rep(counter2) );

    end
elseif Scenario==2 % opposite-wall RIS

    for counter2=indices

        b_cs(counter2)=norm(RIS_xyz-Coordinates2(counter2,:));   % Distance between Scatterer and RIS
        d_cs(counter2)=a_c_rep(counter2)+b_cs(counter2);         % Total distance Tx-Scatterer-RIS

        I_phi=sign(Coordinates2(counter2,2)-y_RIS);   % These are different from Scenario 1
        phi_cs_RIS(counter2)  = I_phi* atand ( abs(y_RIS-Coordinates2(counter2,2) ) / abs(  x_RIS-Coordinates2(counter2,1) ) );

        I_theta=sign(Coordinates2(counter2,3)-z_RIS);  % the same as Scenario 1
        theta_cs_RIS(counter2)=I_theta * asind ( abs (z_RIS-Coordinates2(counter2,3) )/ b_cs(counter2) );

        I_phi_Tx_cs=sign(y_Tx-Coordinates2(counter2,2));
        phi_Tx_cs(counter2) = I_phi_Tx_cs* atand ( abs( Coordinates2(counter2,2)-y_Tx) / abs(Coordinates2(counter2,1)-x_Tx) );

        I_theta_Tx_cs=sign(Coordinates2(counter2,3)-z_Tx);
        theta_Tx_cs(counter2)=I_theta_Tx_cs * asind ( abs (Coordinates2(counter2,3)-z_Tx )/ a_c_rep(counter2) );

    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 4 (Common for Enviroments 1 and 2)
% Array Response Calculation
array_cs_RIS=zeros(sum(S),N);
for counter2=indices
    counter3=1;
    for x=0:sqrt(N)-1
        for y=0:sqrt(N)-1
            array_cs_RIS(counter2,counter3)=exp(1i*k*dis*(x*sind(theta_cs_RIS(counter2)) + y*sind(phi_cs_RIS(counter2))*cosd(theta_cs_RIS(counter2)) )) ;
            counter3=counter3+1;
        end
    end
end

array_Tx_cs=zeros(sum(S),Nt);

if ArrayType == 1
    for counter2 = indices
        counter3=1;
        for x=0:Nt-1
            array_Tx_cs(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_Tx_cs(counter2))*cosd(theta_Tx_cs(counter2)) )) ;
            counter3=counter3+1;
        end
    end

elseif ArrayType == 2
    for counter2 = indices
        counter3=1;
        for x=0:sqrt(Nt)-1
            for y = 0:sqrt(Nt)-1
                array_Tx_cs(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_Tx_cs(counter2))*cosd(theta_Tx_cs(counter2)) + y*sind(theta_Tx_cs(counter2)) )) ;
                counter3=counter3+1;
            end
        end
    end

end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 5 (Common for Enviroments 1 and 2)
% Calculate Link Attenuation and Generate Tx-RIS Channel (h) using 5G Channel Model
% Ge Updated (Version 1.5)
h_NLOS=zeros(N,Nt);
beta=zeros(1,sum(S)); % to be reused for shared clusters (Environment 1)
shadow=beta;          % to be reused for shared clusters (Environment 1) - Version 1.4
for counter2=indices
    X_sigma=randn*sigma_NLOS;
    Lcs_dB=-20*log10(4*pi/lambda) - 10*n_NLOS*(1+b_NLOS*((Frequency-f0)/f0))*log10(d_cs(counter2))- X_sigma;
    Lcs=10^(Lcs_dB/10);
    beta(counter2)=((randn+1i*randn)./sqrt(2));  % common complex gain for shared clusters
    shadow(counter2)=X_sigma;                    % commun shadow factor for shared clusters
    h_NLOS = h_NLOS + beta(counter2)*sqrt(Gain*(cosd(theta_cs_RIS(counter2)))^(2*q))*sqrt(Lcs)*transpose(array_cs_RIS(counter2,:))*array_Tx_cs(counter2,:);  % consider all scatters
end
h_NLOS=h_NLOS.*sqrt(1/M_new);  % normalization
h=h_NLOS+h_LOS; % include the LOS component (if any) h_LOS=0 when there is no LOS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEPS 6-7
% Generation of g (RIS-Rx Channel)
if Environment==1   % GENERATE A LOS CHANNEL (modified in Version 1.4)

    % Version 1.3 (and older) uses the pure LOS channel
    % Version 1.4 uses the 5G LOS component with Practical Path Loss (~1-3 dB less path loss)
    % Version 1.5 ignores Ge^2 in g and includes the element radiation pattern
    % Version 1.6 Calculate array response for ULA at Tx and Rx
    % Version 1.7 Calculate array response for ULA and UPA at Tx and Rx (Random Rx orientation)
    % Calculate Departure Angles Considering RIS and Rx Coordinates
    d_RIS_R=norm(RIS_xyz-Rx_xyz);
    % Elevation Departure Angle
    I_theta=sign(z_Rx - z_RIS);
    theta_Rx_RIS=I_theta * asind( abs(z_Rx-z_RIS)/d_RIS_R ); % AoD of RIS
    % Azimuth Departure Angle
    if Scenario==1
        I_phi=sign(x_RIS - x_Rx);
        phi_Rx_RIS=I_phi * atand( abs(x_Rx-x_RIS)/ abs(y_Rx-y_RIS) ); % AoD of RIS



        % AoA angles of Rx for g_LOS channel in an Indoor
        phi_av_Rx  = rand*180-90;     % mean azimuth
        theta_av_Rx= rand*180-90;      % 均值仰角

        phi_Rx   = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + phi_av_Rx];
        theta_Rx = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + theta_av_Rx];

        %         I_theta_Rx=sign(z_RIS - z_Rx);
        %         theta_Rx=I_theta_Rx * asind( abs(z_Rx-z_RIS)/d_RIS_R ); % AoA of Rx
        %
        %         I_phi_Rx=sign(y_RIS - y_Rx);
        %         phi_Rx =I_phi_Rx * atand( abs(y_Rx-y_RIS)/ abs(x_Rx-x_RIS));

    elseif Scenario==2
        I_phi=sign(y_Rx- y_RIS);
        phi_Rx_RIS=I_phi * atand( abs(y_Rx-y_RIS)/ abs(x_Rx-x_RIS) );


        % AoA angles of Rx for g_LOS channel in an Indoor
        phi_av_Rx  = rand*180-90;     % mean azimuth
        theta_av_Rx= rand*180-90;      % 均值仰角

        phi_Rx   = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + phi_av_Rx];
        theta_Rx = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + theta_av_Rx];

        %         I_theta_Rx=sign(z_RIS - z_Rx);
        %         theta_Rx=I_theta_Rx * asind( abs(z_Rx-z_RIS)/d_RIS_R );   % same as Scenario 1
        %
        %         I_phi_Rx=sign(y_Rx - y_RIS);
        %         phi_Rx =I_phi_Rx * atand( abs(y_Rx-y_RIS)/ abs(x_Rx-x_RIS));
    end
    % Recalculate Array Response for Two Angles (in Rx direction)
    array_2=zeros(1,N);
    counter3=1;
    for x=0:sqrt(N)-1
        for y=0:sqrt(N)-1
            array_2(counter3)=exp(1i*k*dis*(x*sind(theta_Rx_RIS) + y*sind(phi_Rx_RIS)*cosd(theta_Rx_RIS) )) ;
            counter3=counter3+1;
        end
    end

    array_Rx = zeros(1,Nr);
    if ArrayType == 1
        counter3=1;
        for x=0:Nr-1
            array_Rx(counter3)=exp(1i*k*dis*(x*sind(phi_Rx)*cosd(theta_Rx) )) ;
            counter3=counter3+1;
        end
    elseif ArrayType == 2
        counter3=1;
        for x=0:sqrt(Nr)-1
            for y=0:sqrt(Nr)-1
                array_Rx(counter3)=exp(1i*k*dis*(x*sind(phi_Rx)*cosd(theta_Rx)+y*sind(theta_Rx)  )) ;
                counter3=counter3+1;
            end
        end

    end

    % % LOS Link Attenuation (Version 1.3)
    % Ge=pi;
    % L_LOS=(Ge^2 * lambda^2)  / ((4*pi*d_RIS_R)^2);

    L_dB_LOS_2=-20*log10(4*pi/lambda) - 10*n_LOS*(1+b_LOS*((Frequency-f0)/f0))*log10(d_RIS_R)- randn*sigma_LOS;

    L_LOS_2=10^(L_dB_LOS_2/10);
    % Generate g (Pure LOS)
    % g=Ge*sqrt(L_LOS_2)*transpose(array_2)*exp(1i*rand*2*pi);
    g=sqrt(Gain*(cosd(theta_Rx_RIS))^(2*q))*sqrt(L_LOS_2)*transpose(array_2)*array_Rx*exp(1i*rand*2*pi);   % Version 1.5

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % GENERATION OF g FOR OUTDOORS (Involves STEPS 1-5)
    % including a LOS component (Version 1.4)

    % For simplicity in algorithm, we use "_2" for this link compared to Tx-RIS

elseif Environment==2

    d_RIS_R=norm(RIS_xyz-Rx_xyz);

    %%%%%%%%%%%%%%%% LOS COMPONENT CALCULATION (Version 1.4) from Step 1

    % UMi LOS Probability
    p_LOS_2=min([20/d_RIS_R,1])*(1-exp(-d_RIS_R/39)) + exp(-d_RIS_R/39);

    I_LOS_2=randsrc(1,1,[1,0;p_LOS_2 1-p_LOS_2]);


    if I_LOS_2==1
        %%%%%%%%%%%%
        if Scenario==1   % side-wall RIS


            I_phi=sign(x_RIS-x_Rx);
            phi_RIS_R_LOS = I_phi* atand ( abs( x_RIS-x_Rx) / abs(y_RIS-y_Rx) );

            I_theta=sign(z_Rx-z_RIS);
            theta_RIS_R_LOS=I_theta * asind ( abs (z_RIS-z_Rx )/ d_RIS_R );


            % AoA angles of Rx for g_LOS channel in an Outdoor
            phi_av_Rx_LOS = rand*180-90;     % mean azimuth
            theta_av_Rx_LOS= rand*180-90;      % 均值仰角

            phi_Rx_LOS   = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + phi_av_Rx_LOS];
            theta_Rx_LOS = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + theta_av_Rx_LOS];

            %             I_phi_Rx_LOS=sign(y_RIS-y_Rx);
            %             phi_Rx_LOS = I_phi_Rx_LOS* atand ( abs( y_RIS-y_Rx) / abs(x_RIS-x_Rx) );
            %
            %             I_theta_Rx_LOS=sign(z_RIS-z_Rx);
            %             theta_Rx_LOS=I_theta_Rx_LOS * asind ( abs (z_RIS-z_Rx )/ d_RIS_R );


        elseif Scenario==2 % opposite-wall RIS


            I_phi=sign(y_Rx-y_RIS);   % These are different from Scenario 1
            phi_RIS_R_LOS  = I_phi* atand ( abs(y_RIS-y_Rx ) / abs(  x_RIS-x_Rx ) );

            I_theta=sign(z_Rx-z_RIS);  % Same as Scenario 1
            theta_RIS_R_LOS=I_theta * asind ( abs (z_RIS-z_Rx )/ d_RIS_R );


            % AoA angles of Rx for g_LOS channel in an Outdoor
            phi_av_Rx_LOS = rand*180-90;     % mean azimuth
            theta_av_Rx_LOS= rand*180-90;      % 均值仰角

            phi_Rx_LOS   = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + phi_av_Rx_LOS];
            theta_Rx_LOS = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + theta_av_Rx_LOS];

            %             I_phi_Rx_LOS=sign(y_Rx-y_RIS);
            %             phi_Rx_LOS = I_phi_Rx_LOS* atand ( abs( y_RIS-y_Rx) / abs(x_RIS-x_Rx) );
            %
            %             I_theta_Rx_LOS=sign(z_RIS-z_Rx);
            %             theta_Rx_LOS=I_theta_Rx_LOS * asind ( abs (z_RIS-z_Rx )/ d_RIS_R );
            %

        end


        % Array Response Calculation (LOS)
        array_RIS_Rx_LOS=zeros(1,N);
        array_Rx_LOS = zeros(1,Nr);


        counter3=1;
        for x=0:sqrt(N)-1
            for y=0:sqrt(N)-1
                array_RIS_Rx_LOS(counter3)=exp(1i*k*dis*(x*sind(theta_RIS_R_LOS) + y*sind(phi_RIS_R_LOS)*cosd(theta_RIS_R_LOS) )) ;
                counter3=counter3+1;
            end
        end

        if ArrayType == 1
            counter3=1;
            for x=0:Nr-1
                array_Rx_LOS(counter3)=exp(1i*k*dis*(x*sind(phi_Rx_LOS)*cosd(theta_Rx_LOS) )) ;
                counter3=counter3+1;
            end

        elseif ArrayType == 2
            counter3=1;
            for x = 0:sqrt(Nr)-1
                for y = 0:sqrt(Nr)-1
                    array_Rx_LOS(counter3)=exp(1i*k*dis*(x*sind(phi_Rx_LOS)*cosd(theta_Rx_LOS)+y*sind(theta_Rx_LOS))) ;
                    counter3=counter3+1;
                end
            end

        end
        % Link Attentuation (LOS)



        L_dB_LOS_2=-20*log10(4*pi/lambda) - 10*n_LOS*(1+b_LOS*((Frequency-f0)/f0))*log10(d_RIS_R)- randn*sigma_LOS;

        L_LOS_2=10^(L_dB_LOS_2/10);

        % LOS COMPONENT GENERATED
        g_LOS=sqrt(L_LOS_2)*transpose(array_RIS_Rx_LOS)*array_Rx_LOS*exp(1i*rand*2*pi)*sqrt(Gain*(cosd(theta_RIS_R_LOS))^(2*q));


        %%%%%%%%%%%%
    else
        g_LOS=0;
    end





    %%%%%%%%%%%%%%%%

    % Generate New Clusters/Scatters (Step 2) - Ensure that All Scattters are above ground
    for generate2=1:100  % To ensure that at least one scatterer exist

        % lambda_p was defined before for 28/73 GHz
        C_2=max([1,poissrnd(lambda_p)]);  % Poisson distributed

        % Number of Sub-rays per Cluster
        S_2=randi(30,1,C_2);
        % Azimuth/Elevation Departure Angles
        phi_Tx_2=[ ];
        theta_Tx_2=[ ];
        phi_av_2=zeros(1,C_2);
        theta_av_2=zeros(1,C_2);

        for counter=1:C_2

            phi_av_2(counter)  = rand*90-45;   % mean azimuth (reduced to ensure that all scatters are within the field of view)
            theta_av_2(counter)= rand*90-45; % 均值仰角  (needs update & most of the clusters are above ceiling)
            % for outdoors this might not be a big concern
            % cluster angles: First S(1) belongs to Cluster 1, Next S(2) belongs to Cluster 2....
            phi_Tx_2   = [phi_Tx_2,  log(rand(1,S_2(counter))./rand(1,S_2(counter)))*sqrt(25/2) + phi_av_2(counter)];
            theta_Tx_2 = [theta_Tx_2,log(rand(1,S_2(counter))./rand(1,S_2(counter)))*sqrt(25/2) + theta_av_2(counter)];

        end

        % Cluster Distances
        a_c_2=1+rand(1,C_2)*(d_RIS_R-1);    % Cluster distances uniform [1,d_RIS_R] % can be modified later


        % Reducing Distances for Outside Clusters - Check Cluster Coordinates with Mean Angles

        Coordinates_2=zeros(C_2,3);          % for Clusters
        Coordinates2_2=zeros(sum(S_2),3);    % for Scatterers

        for counter=1:C_2
            loop=1;
            if Scenario==1       % CHECK !!!
                Coordinates_2(counter,:)=[x_RIS - a_c_2(counter)*cosd(theta_av_2(counter))*sind(phi_av_2(counter)),...
                    y_RIS - a_c_2(counter)*cosd(theta_av_2(counter))*cosd(phi_av_2(counter)),...
                    z_RIS + a_c_2(counter)*sind(theta_av_2(counter))];

            elseif Scenario==2   % CHECK !!!
                Coordinates_2(counter,:)=[x_RIS - a_c_2(counter)*cosd(theta_av_2(counter))*cosd(phi_av_2(counter)),...
                    y_RIS + a_c_2(counter)*cosd(theta_av_2(counter))*sind(phi_av_2(counter)),...
                    z_RIS + a_c_2(counter)*sind(theta_av_2(counter))];

            end

            while  Coordinates_2(counter,3)<0    % only underground clusters will be ignored
                a_c_2(counter)=    0.8*a_c_2(counter)  ;     % reduce distance 10%-20% if coordinates are not met!

                % Note: While the above 10% reduction ensures that all clusters are in the range, to ensure that most of the scatterers are
                % in the range, we may consider 20% or 30% reduction.

                if Scenario==1       % CHECK !!!
                    Coordinates_2(counter,:)=[x_RIS - a_c_2(counter)*cosd(theta_av_2(counter))*sind(phi_av_2(counter)),...
                        y_RIS - a_c_2(counter)*cosd(theta_av_2(counter))*cosd(phi_av_2(counter)),...
                        z_RIS + a_c_2(counter)*sind(theta_av_2(counter))];

                elseif Scenario==2   % CHECK !!!
                    Coordinates_2(counter,:)=[x_RIS - a_c_2(counter)*cosd(theta_av_2(counter))*cosd(phi_av_2(counter)),...
                        y_RIS + a_c_2(counter)*cosd(theta_av_2(counter))*sind(phi_av_2(counter)),...
                        z_RIS + a_c_2(counter)*sind(theta_av_2(counter))];
                end
                %    loop=loop+1

            end

            %     % Plot Clusters (comment figures in mass generation of channels (with repeat))
            % plot3(Coordinates_2(counter,1),Coordinates_2(counter,2),Coordinates_2(counter,3),'bdiamond')
            % hold on;
        end

        % Plot Scatterers
        a_c_rep_2=[];
        for counter3=1:C_2
            a_c_rep_2=[a_c_rep_2,repmat(a_c_2(counter3),1,S_2(counter3))];
        end


        for counter2=1:sum(S_2)
            if Scenario==1       % CHECK !!!
                Coordinates2_2(counter2,:)=[x_RIS - a_c_rep_2(counter2)*cosd(theta_Tx_2(counter2))*sind(phi_Tx_2(counter2)),...
                    y_RIS - a_c_rep_2(counter2)*cosd(theta_Tx_2(counter2))*cosd(phi_Tx_2(counter2)),...
                    z_RIS + a_c_rep_2(counter2)*sind(theta_Tx_2(counter2))];

            elseif Scenario==2   % CHECK !!!
                Coordinates2_2(counter2,:)=[x_RIS - a_c_rep_2(counter2)*cosd(theta_Tx_2(counter2))*cosd(phi_Tx_2(counter2)),...
                    y_RIS + a_c_rep_2(counter2)*cosd(theta_Tx_2(counter2))*sind(phi_Tx_2(counter2)),...
                    z_RIS + a_c_rep_2(counter2)*sind(theta_Tx_2(counter2))];

            end


            %       % comment figures in mass generation of channels (with repeat)
            %     plot3(Coordinates2_2(counter2,1),Coordinates2_2(counter2,2),Coordinates2_2(counter2,3),'b.')
            %   hold on;

        end



        % Correction on Scatters

        ignore_2=[];

        for counter2=1:sum(S_2)
            if  Coordinates2_2(counter2,3)<0   % only underground scatterers
                ignore_2=[ignore_2,counter2];   % contains the indices of scatterers that can be ignored
            end
        end

        % updated indices
        indices_2=setdiff(1:sum(S_2),ignore_2);    % the set of active scatterer indices
        M_new_2=length(indices_2);               % number of IOs inside the room or above ground


        % Neccessary Loop to have at least one scatterer
        if M_new_2>0 % all scatters are outside
            break  % break generate=1:100
        end

    end  % for generate=1:100


    % Calculate Array Response (Step 4)
    % STEP 4
    % Array Response Calculation
    array_2=zeros(sum(S_2),N);

    for counter2=indices_2
        counter3=1;
        for x=0:sqrt(N)-1
            for y=0:sqrt(N)-1

                array_2(counter2,counter3)=exp(1i*k*dis*(x*sind(theta_Tx_2(counter2)) + y*sind(phi_Tx_2(counter2))*cosd(theta_Tx_2(counter2)) )) ;
                counter3=counter3+1;
            end
        end
    end


    % Calculate Link Lengths (Step 3) and Path Loss (Step 5)
    b_cs_2=zeros(1,sum(S_2));
    d_cs_2=zeros(1,sum(S_2));
    phi_cs_Rx = zeros(1,sum(S_2));
    theta_cs_Rx = zeros(1,sum(S_2));
    for counter2=indices_2

        b_cs_2(counter2)=norm(Rx_xyz-Coordinates2_2(counter2,:));   % Distance between Scatterer and Rx
        d_cs_2(counter2)=a_c_rep_2(counter2)+b_cs_2(counter2);         % Total distance RIS-Scatterer-Rx

        % AoA angles of Rx for NLOS RIS-Rx channel in an Outdoor
        if Scenario == 1

            phi_av_Rx_NLOS(counter2)  = rand*180-90;     % mean azimuth
            theta_av_Rx_NLOS(counter2)= rand*180-90;      % 均值仰角

            phi_cs_Rx(counter2)   = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + phi_av_Rx_NLOS(counter2)];
            theta_cs_Rx(counter2) = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + theta_av_Rx_NLOS(counter2)];

            %             I_phi_cs_Rx=sign(Coordinates2_2(counter2,2)-y_Rx);
            %             phi_cs_Rx(counter2) = I_phi_cs_Rx* atand ( abs( Coordinates2_2(counter2,2)-y_Rx) / abs(Coordinates2_2(counter2,1)-x_Rx) );
            %
            %             I_theta_cs_Rx=sign(Coordinates2_2(counter2,3)-z_Rx);
            %             theta_cs_Rx(counter2)=I_theta_cs_Rx * asind ( abs (Coordinates2_2(counter2,3)-z_Rx )/ b_cs_2(counter2) );
        elseif Scenario ==2

            phi_av_Rx_NLOS(counter2)  = rand*180-90;     % mean azimuth
            theta_av_Rx_NLOS(counter2)= rand*180-90;      % 均值仰角

            phi_cs_Rx(counter2)   = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + phi_av_Rx_NLOS(counter2) ];
            theta_cs_Rx(counter2)  = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + theta_av_Rx_NLOS(counter2) ];


            %             I_phi_cs_Rx=sign(y_Rx-Coordinates2_2(counter2,2));
            %             phi_cs_Rx(counter2) = I_phi_cs_Rx* atand ( abs( Coordinates2_2(counter2,2)-y_Rx) / abs(Coordinates2_2(counter2,1)-x_Rx) );
            %
            %             I_theta_cs_Rx=sign(Coordinates2_2(counter2,3)-z_Rx);
            %             theta_cs_Rx(counter2)=I_theta_cs_Rx * asind ( abs (Coordinates2_2(counter2,3)-z_Rx )/ b_cs_2(counter2) );
        end
    end

    array_Rx_cs=zeros(sum(S_2),Nr);

    if ArrayType == 1
        for counter2 = indices_2
            counter3=1;
            for x=0:Nr-1
                array_Rx_cs(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Rx(counter2))*cosd(theta_cs_Rx(counter2)) )) ;
                counter3=counter3+1;
            end
        end

    elseif ArrayType == 2
        for counter2 = indices_2
            counter3=1;
            for x=0:sqrt(Nr)-1
                for  y=0:sqrt(Nr)-1
                    array_Rx_cs(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Rx(counter2))*cosd(theta_cs_Rx(counter2))+y*sind(theta_cs_Rx(counter2)) )) ;
                    counter3=counter3+1;
                end
            end
        end

    end
    %% 生成 g（步骤 5）
    g_NLOS=zeros(N,Nr);

    for counter2=indices_2

        X_sigma_2=randn*sigma_NLOS;

        Lcs_dB_2=-20*log10(4*pi/lambda) - 10*n_NLOS*(1+b_NLOS*((Frequency-f0)/f0))*log10(d_cs_2(counter2))- X_sigma_2;

        Lcs_2=10^(Lcs_dB_2/10);

        beta_2=((randn+1i*randn)./sqrt(2));

        g_NLOS = g_NLOS + beta_2*sqrt(Lcs_2)*sqrt(Gain*(cosd(theta_Tx_2(counter2)))^(2*q))*transpose(array_2(counter2,:))*array_Rx_cs(counter2,:);   % 包含或更新 RIS 元素的增益
    end

    g_NLOS=g_NLOS.*sqrt(1/M_new_2);  % normalization

    % for Environment 2 only (Enviroment 1 is already LOS)
    g=g_NLOS+g_LOS;


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 8
% Generation of h_SISO

if Environment==1  % WITH SHARED CLUSTERS AND COMMON LOS

    d_T_R=norm(Tx_xyz-Rx_xyz);

    d_cs_tilde=zeros(1,sum(S));
    h_SISO_NLOS=0;

    for counter2=indices

        % due to shared clusters d_cs_tilde ~ d_cs
        d_cs_tilde(counter2) = a_c_rep(counter2) + norm(Coordinates2(counter2,:)- Rx_xyz);


        I_phi_Tx_cs_SISO=sign(y_Tx-Coordinates2(counter2,2));
        phi_Tx_cs_SISO(counter2) = I_phi_Tx_cs_SISO* atand ( abs( Coordinates2(counter2,2)-y_Tx) / abs(Coordinates2(counter2,1)-x_Tx) );

        I_theta_Tx_cs_SISO=sign(Coordinates2(counter2,3)-z_Tx);
        theta_Tx_cs_SISO(counter2)=I_theta_Tx_cs_SISO * asind ( abs (Coordinates2(counter2,3)-z_Tx )/ a_c_rep(counter2) );

        % AoA for Rx in an Indoor
        phi_av_SISO(counter2)  = rand*180-90;     % mean azimuth
        theta_av_SISO(counter2)= rand*180-90;      % 均值仰角

        phi_cs_Rx_SISO(counter2)   = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + phi_av_SISO(counter2)];
        theta_cs_Rx_SISO(counter2) = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + theta_av_SISO(counter2)];

        %
        %         I_phi_cs_Rx_SISO=sign(Coordinates2(counter2,2)-y_Rx);
        %         phi_cs_Rx_SISO(counter2)  = I_phi_cs_Rx_SISO* atand ( abs( Coordinates2(counter2,2)-y_Rx) / abs(Coordinates2(counter2,1)-x_Rx) );
        %
        %         I_theta_cs_Rx_SISO=sign(Coordinates2(counter2,3)-z_Rx);
        %         theta_cs_Rx_SISO(counter2)=I_theta_cs_Rx_SISO * asind ( abs (z_RIS-Coordinates2(counter2,3) )/ norm(Coordinates2(counter2,:)- Rx_xyz) );
        %
        if ArrayType == 1
            counter3=1;
            for x=0:Nr-1
                array_Rx_cs_SISO(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Rx_SISO(counter2))*cosd(theta_cs_Rx_SISO(counter2)) )) ;
                counter3=counter3+1;
            end

            counter3=1;
            for x=0:Nt-1
                array_Tx_cs_SISO(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_Tx_cs_SISO(counter2))*cosd(theta_Tx_cs_SISO(counter2)) )) ;
                counter3=counter3+1;
            end

        elseif ArrayType ==2

            counter3=1;
            for x=0:sqrt(Nr)-1
                for y=0:sqrt(Nr)-1
                    array_Rx_cs_SISO(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Rx_SISO(counter2))*cosd(theta_cs_Rx_SISO(counter2))+y*sind(theta_cs_Rx_SISO(counter2)) )) ;
                    counter3=counter3+1;
                end
            end

            counter3=1;
            for x=0:sqrt(Nt)-1
                for y=0:sqrt(Nt)-1
                    array_Tx_cs_SISO(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_Tx_cs_SISO(counter2))*cosd(theta_Tx_cs_SISO(counter2))+y*sind(theta_Tx_cs_SISO(counter2))  )) ;
                    counter3=counter3+1;
                end
            end

        end

        %    X_sigma=randn*sigma; SHARED CLUSTERS (same Shadow Factor - Version 1.4 Updated)

        Lcs_dB_SISO=-20*log10(4*pi/lambda) - 10*n_NLOS*(1+b_NLOS*((Frequency-f0)/f0))*log10(d_cs_tilde(counter2))- shadow(counter2);

        Lcs_SISO=10^(Lcs_dB_SISO/10);

        % We consider the same complex path gain (small scale fading) with an excess phase (similar to array response)
        eta=k* ( norm(Coordinates2(counter2,:)- RIS_xyz) -  norm(Coordinates2(counter2,:)- Rx_xyz));

        h_SISO_NLOS = h_SISO_NLOS + beta(counter2)*exp(1i*eta)*sqrt(Lcs_SISO)*transpose(array_Rx_cs_SISO(counter2,:))*array_Tx_cs_SISO(counter2,:);
    end

    h_SISO_NLOS=h_SISO_NLOS.*sqrt(1/M_new);  % normalization

    if z_RIS >= z_Tx

        % % Include LOS component (Version 1.4)
        %     % InH LOS Probability
        if d_T_R<= 1.2
            p_LOS_3=1;
        elseif 1.2<d_T_R && d_T_R<6.5
            p_LOS_3=exp(-(d_T_R-1.2)/4.7);
        else
            p_LOS_3=0.32*exp(-(d_T_R-6.5)/32.6);
        end

        I_LOS_3=randsrc(1,1,[1,0;p_LOS_3 1-p_LOS_3]);

        % Do not recalculate, if T-RIS has LOS, we might have LOS for h_SISO as well (for ground level RIS)
        % If we have LOS for Tx-RIS, we have LOS for Tx-Rx
    elseif z_RIS < z_Tx  % RIS in the ground level

        I_LOS_3=I_LOS;

    end

    if I_LOS_3==1
        L_SISO_LOS_dB=-20*log10(4*pi/lambda) - 10*n_LOS*(1+b_LOS*((Frequency-f0)/f0))*log10(d_T_R)- randn*sigma_LOS;
        L_SISO_LOS=10^(L_SISO_LOS_dB/10);

        I_phi_Tx_SISO=sign(y_Tx-y_Rx);
        phi_Tx_SISO = I_phi_Tx_SISO* atand ( abs( y_Tx-y_Rx) / abs(x_Tx-x_Rx) );

        I_theta_Tx_SISO=sign(z_Rx-z_Tx);
        theta_Tx_SISO= I_theta_Tx_SISO* atand ( abs( z_Rx-z_Tx) / abs(d_T_R) );


        % AoA of Rx for Tx-Rx channel in an Indoor
        phi_av_SISO_LOS = rand*180-90;     % mean azimuth
        theta_av_SISO_LOS= rand*180-90;      % 均值仰角

        phi_Rx_SISO  = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + phi_av_SISO_LOS];
        theta_Rx_SISO= [log(rand(1,1)./rand(1,1))*sqrt(25/2) + theta_av_SISO_LOS];

        %         I_phi_Rx_SISO=sign(y_Tx-y_Rx);
        %         phi_Rx_SISO = I_phi_Rx_SISO* atand ( abs( y_Tx-y_Rx) / abs(x_Tx-x_Rx) );
        %
        %         I_theta_Rx_SISO=sign(z_Tx-z_Rx);
        %         theta_Rx_SISO= I_theta_Rx_SISO* atand ( abs( z_Rx-z_Tx) / abs(d_T_R) );



        if ArrayType == 1
            counter3=1;
            for x=0:Nt-1
                array_Tx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Tx_SISO)*cosd(theta_Tx_SISO) )) ;
                counter3=counter3+1;
            end

            counter3=1;
            for x=0:Nr-1
                array_Rx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Rx_SISO)*cosd(theta_Rx_SISO) )) ;
                counter3=counter3+1;
            end

        elseif ArrayType == 2
            counter3=1;
            for x=0:sqrt(Nt)-1
                for y=0:sqrt(Nt)-1
                    array_Tx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Tx_SISO)*cosd(theta_Tx_SISO)+y*sind(theta_Tx_SISO)  )) ;
                    counter3=counter3+1;
                end
            end

            counter3=1;
            for x=0:sqrt(Nr)-1
                for y=0:sqrt(Nr)-1
                    array_Rx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Rx_SISO)*cosd(theta_Rx_SISO) +y*sind(theta_Rx_SISO))) ;
                    counter3=counter3+1;
                end
            end

        end

        h_SISO_LOS= sqrt(L_SISO_LOS)*exp(1i*rand*2*pi)*transpose(array_Rx_SISO)*array_Tx_SISO;
    else
        h_SISO_LOS=0;
    end

    h_SISO=h_SISO_NLOS + h_SISO_LOS; % include LOS Component if I_LOS_3==1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif Environment==2 % WITH NEW CLUSTERS

    % Generate New Clusters/Scatterers (Outdoors)
    d_T_R=norm(Tx_xyz-Rx_xyz);
    % Calculate LOS Component (Version 1.4)
    % UMi LOS Probability
    p_LOS_3=min([20/d_T_R,1])*(1-exp(-d_T_R/39)) + exp(-d_T_R/39);
    I_LOS_3=randsrc(1,1,[1,0;p_LOS_3 1-p_LOS_3]);
    if I_LOS_3==1
        L_SISO_LOS_dB=-20*log10(4*pi/lambda) - 10*n_LOS*(1+b_LOS*((Frequency-f0)/f0))*log10(d_T_R)- randn*sigma_LOS;
        L_SISO_LOS=10^(L_SISO_LOS_dB/10);
        I_phi_Tx_SISO=sign(y_Tx-y_Rx);
        phi_Tx_SISO = I_phi_Tx_SISO* atand ( abs( y_Tx-y_Rx) / abs(x_Tx-x_Rx) );

        I_theta_Tx_SISO=sign(z_Rx-z_Tx);
        theta_Tx_SISO= I_theta_Tx_SISO* atand ( abs( z_Rx-z_Tx) / abs(d_T_R) );


        % AoA of Rx for Tx-Rx channel in an Outdoor
        phi_av_SISO_LOS = rand*180-90;     % mean azimuth
        theta_av_SISO_LOS= rand*180-90;      % 均值仰角

        phi_Rx_SISO  = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + phi_av_SISO_LOS];
        theta_Rx_SISO= [log(rand(1,1)./rand(1,1))*sqrt(25/2) + theta_av_SISO_LOS];

        %         I_phi_Rx_SISO=sign(y_Tx-y_Rx);
        %         phi_Rx_SISO = I_phi_Rx_SISO* atand ( abs( y_Tx-y_Rx) / abs(x_Tx-x_Rx) );
        %
        %         I_theta_Rx_SISO=sign(z_Tx-z_Rx);
        %         theta_Rx_SISO= I_theta_Rx_SISO* atand ( abs( z_Rx-z_Tx) / abs(d_T_R) );



        if ArrayType == 1
            counter3=1;
            for x=0:Nt-1
                array_Tx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Tx_SISO)*cosd(theta_Tx_SISO) )) ;
                counter3=counter3+1;
            end

            counter3=1;
            for x=0:Nr-1
                array_Rx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Rx_SISO)*cosd(theta_Rx_SISO) )) ;
                counter3=counter3+1;
            end

        elseif ArrayType == 2
            counter3=1;
            for x=0:sqrt(Nt)-1
                for y=0:sqrt(Nt)-1
                    array_Tx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Tx_SISO)*cosd(theta_Tx_SISO)+y*sind(theta_Tx_SISO) )) ;
                    counter3=counter3+1;
                end
            end

            counter3=1;
            for x=0:sqrt(Nr)-1
                for y=0:sqrt(Nr)-1
                    array_Rx_SISO(counter3)=exp(1i*k*dis*(x*sind(phi_Rx_SISO)*cosd(theta_Rx_SISO)+y*sind(theta_Rx_SISO) )) ;
                    counter3=counter3+1;
                end
            end

        end


        h_SISO_LOS= sqrt(L_SISO_LOS)*exp(1i*rand*2*pi)*transpose(array_Rx_SISO)*array_Tx_SISO;;   % LOS Component for the Tx-Rx link
    else
        h_SISO_LOS=0;
    end
    % Generate New Clusters/Scatters (Step 2) - Ensure that All Scattters are above ground
    for generate3=1:100  % To ensure that at least one scatterer exist
        % lambda_p was defined before for 28/73 GHz
        C_3=max([1,poissrnd(lambda_p)]);  % Poisson distributed
        % Number of Sub-rays per Cluster
        S_3=randi(30,1,C_3);
        % Azimuth/Elevation Departure Angles
        phi_Tx_3=[ ];
        theta_Tx_3=[ ];
        phi_av_3=zeros(1,C_3);
        theta_av_3=zeros(1,C_3);
        for counter=1:C_3
            phi_av_3(counter)  = rand*180-90;   % mean azimuth
            theta_av_3(counter)= rand*90-45; % 均值仰角  (needs update & most of the clusters are above ceiling)
            % for outdoors this might not be a big concern
            % cluster angles: First S(1) belongs to Cluster 1, Next S(2) belongs to Cluster 2....
            phi_Tx_3   = [phi_Tx_3,  log(rand(1,S_3(counter))./rand(1,S_3(counter)))*sqrt(25/2) + phi_av_3(counter)];
            theta_Tx_3 = [theta_Tx_3,log(rand(1,S_3(counter))./rand(1,S_3(counter)))*sqrt(25/2) + theta_av_3(counter)];
        end
        % Cluster Distances
        a_c_3=1+rand(1,C_3)*(d_T_R-1);    % Cluster distances uniform [1,d_T_R]
        % Reducing Distances for Outside Clusters - Check Cluster Coordinates with Mean Angles
        Coordinates_3=zeros(C_3,3);          % for Clusters
        Coordinates2_3=zeros(sum(S_3),3);    % for Scatterers
        for counter=1:C_3
            loop=1;
            Coordinates_3(counter,:)=[x_Tx + a_c_3(counter)*cosd(theta_av_3(counter))*cosd(phi_av_3(counter)),...
                y_Tx - a_c_3(counter)*cosd(theta_av_3(counter))*sind(phi_av_3(counter)),...
                z_Tx + a_c_3(counter)*sind(theta_av_3(counter))] ;
            while  Coordinates_3(counter,3)<0    % only underground clusters will be ignored
                a_c_3(counter)=    0.8*a_c_3(counter)  ;     % reduce distance 10%-20% if coordinates are not met!
                % Note: While the above 10% reduction ensures that all clusters are in the range, to ensure that most of the scatterers are
                % in the range, we may consider 20% or 30% reduction.

                Coordinates_3(counter,:)=[x_Tx + a_c_3(counter)*cosd(theta_av_3(counter))*cosd(phi_av_3(counter)),...
                    y_Tx - a_c_3(counter)*cosd(theta_av_3(counter))*sind(phi_av_3(counter)),...
                    z_Tx + a_c_3(counter)*sind(theta_av_3(counter))] ;
                %    loop=loop+1
            end
            %     % Plot Clusters (comment figures in mass generation of channels (with repeat))
            % plot3(Coordinates_3(counter,1),Coordinates_3(counter,2),Coordinates_3(counter,3),'rdiamond')
            % hold on;
        end
        % Plot Scatterers
        a_c_rep_3=[];
        for counter3=1:C_3
            a_c_rep_3=[a_c_rep_3,repmat(a_c_3(counter3),1,S_3(counter3))];
        end
        for counter2=1:sum(S_3)
            Coordinates2_3(counter2,:)=[x_Tx + a_c_rep_3(counter2)*cosd(theta_Tx_3(counter2))*cosd(phi_Tx_3(counter2)),...
                y_Tx - a_c_rep_3(counter2)*cosd(theta_Tx_3(counter2))*sind(phi_Tx_3(counter2)),...
                z_Tx + a_c_rep_3(counter2)*sind(theta_Tx_3(counter2))] ;
            %       % comment figures in mass generation of channels (with repeat)
            %     plot3(Coordinates2_3(counter2,1),Coordinates2_3(counter2,2),Coordinates2_3(counter2,3),'r.')
            %   hold on;
        end
        % Correction on Scatters
        ignore_3=[];
        for counter2=1:sum(S_3)
            if  Coordinates2_3(counter2,3)<0   % only underground scatterers
                ignore_3=[ignore_3,counter2];   % contains the indices of scatterers that can be ignored
            end
        end
        % updated indices
        indices_3=setdiff(1:sum(S_3),ignore_3);    % the set of active scatterer indices
        M_new_3=length(indices_3);               % number of IOs inside the room or above ground
        % Neccessary Loop to have at least one scatterer
        if M_new_3>0 % all scatters are outside
            break  % break generate=1:100
        end
    end  % for generate=1:100
    % Calculate Link Lengths and Path Loss
    b_cs_3=zeros(1,sum(S_3));
    d_cs_3=zeros(1,sum(S_3));
    for counter2=indices_3
        b_cs_3(counter2)=norm(Tx_xyz-Coordinates2_3(counter2,:));      % Distance between Scatterer and Tx
        d_cs_3(counter2)=a_c_rep_3(counter2)+b_cs_3(counter2);         % Total distance Tx-Scatterer-Rx


        %%
        I_phi_Tx2=sign(y_Tx-Coordinates2_3(counter2,2));
        phi_cs_Tx2(counter2) = I_phi_Tx2* atand ( abs( Coordinates2_3(counter2,2)-y_Tx) / abs(Coordinates2_3(counter2,1)-x_Tx) );

        I_theta_Tx2=sign(Coordinates2_3(counter2,3)-z_Tx);
        theta_cs_Tx2(counter2)=I_theta_Tx2 * asind ( abs (Coordinates2_3(counter2,3)-z_Tx )/ a_c_rep_3(counter2) );


        %% AoA of Rx for NLOS Tx-Rx Channel in an Outdoor
        phi_av_cs_Rx2(counter2) = rand*180-90;     % mean azimuth
        theta_av_cs_Rx2(counter2) = rand*180-90;      % 均值仰角

        phi_cs_Rx2(counter2)  = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + phi_av_cs_Rx2(counter2) ];
        theta_cs_Rx2(counter2) = [log(rand(1,1)./rand(1,1))*sqrt(25/2) + theta_av_cs_Rx2(counter2) ];

        %             I_phi_Rx2=sign(Coordinates2_3(counter2,2)-y_Rx);
        %             phi_cs_Rx2(counter2) = I_phi_Rx2* atand ( abs( Coordinates2_3(counter2,2)-y_Rx) / abs(Coordinates2_3(counter2,1)-x_Rx) );
        %
        %             I_theta_Rx2=sign(Coordinates2_3(counter2,3)-z_Rx);
        %             theta_cs_Rx2(counter2)=I_theta_Rx2 * asind ( abs (Coordinates2_3(counter2,3)-z_Rx )/b_cs_3(counter2) );
        %


        %%


    end

    if ArrayType == 1
        array_Rx_cs2=zeros(sum(S_3),Nr);
        for counter2 = indices_3
            counter3=1;
            for x=0:Nr-1
                array_Rx_cs2(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Rx2(counter2))*cosd(theta_cs_Rx2(counter2)) )) ;
                counter3=counter3+1;
            end
        end

        array_Tx_cs2=zeros(sum(S_3),Nt);


        for counter2 = indices_3
            counter3=1;
            for x=0:Nt-1
                array_Tx_cs2(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Tx2(counter2))*cosd(theta_cs_Tx2(counter2)) )) ;
                counter3=counter3+1;
            end
        end

    elseif ArrayType == 2

        array_Rx_cs2=zeros(sum(S_3),Nr);
        for counter2 = indices_3
            counter3=1;
            for x=0:sqrt(Nr)-1
                for y=0:sqrt(Nr)-1
                    array_Rx_cs2(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Rx2(counter2))*cosd(theta_cs_Rx2(counter2))+y*sind(theta_cs_Rx2(counter2)) )) ;
                    counter3=counter3+1;
                end
            end
        end

        array_Tx_cs2=zeros(sum(S_3),Nt);


        for counter2 = indices_3
            counter3=1;
            for x=0:sqrt(Nt)-1
                for y=0:sqrt(Nt)-1
                    array_Tx_cs2(counter2,counter3)=exp(1i*k*dis*(x*sind(phi_cs_Tx2(counter2))*cosd(theta_cs_Tx2(counter2))+y*sind(theta_cs_Tx2(counter2)) )) ;
                    counter3=counter3+1;
                end
            end
        end


    end
    % 生成 h_SISO
    h_SISO_NLOS=0;
    for counter2=indices_3
        X_sigma_3=randn*sigma_NLOS;
        Lcs_dB_SISO_3=-20*log10(4*pi/lambda) - 10*n_NLOS*(1+b_NLOS*((Frequency-f0)/f0))*log10(d_cs_3(counter2))- X_sigma_3;
        Lcs_SISO_3=10^(Lcs_dB_SISO_3/10);
        h_SISO_NLOS = h_SISO_NLOS + ((randn+1i*randn)/sqrt(2))*sqrt(Lcs_SISO_3)*transpose(array_Rx_cs2(counter2,:))*array_Tx_cs2(counter2,:);
    end
    h_SISO_NLOS = h_SISO_NLOS.*sqrt(1/M_new_3);  % normalization
    h_SISO = h_SISO_NLOS + h_SISO_LOS;    % h_SISO_LOS=0  if there is no LOS
end
end
