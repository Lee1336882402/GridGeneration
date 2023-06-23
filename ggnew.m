Mx=121;%需要是奇数
My=101;
tol=10^(-5);
X=ones(My,Mx);Y=ones(My,Mx);
lambda = 0;
%定义边界初值
%定义翼型上的结点x坐标
syms x
y=0.594689181*(0.298222773*sqrt(x+0.5) - 0.127125232*(x+0.5) - 0.357907906*(x+0.5)^2 + 0.291984971*(x+0.5)^3 - 0.105174606*(x+0.5)^4);
theta=linspace(0,2*pi,Mx);
pnt=0.5;
X_boundary=zeros(1,Mx);
Y_boundary=zeros(1,Mx);
X_boundary(1)=0.5;X_boundary(Mx)=0.5;X_boundary((Mx+1)/2)=-0.5;
Y_boundary(1)=0;Y_boundary(Mx)=0;Y_boundary((Mx+1)/2)=0;
% for i = 1:(Mx-3)/2
%     epspnt=0.001;
%     while vpa(subs(y,x,pnt)/pnt)<tan(theta(1+i))||pnt*tan(theta(1+i))<0
%         pnt=pnt-epspnt;
%     end
%     X_boundary(1+i)=pnt;
%     Y_boundary(1+i)=subs(y,x,pnt);
% end
% for i = 1:(Mx-3)/2
%     X_boundary((Mx+1)/2+i)=X_boundary((Mx+1)/2-i);
%     Y_boundary((Mx+1)/2+i)=-subs(y,x,X_boundary((Mx+1)/2+i));
% end
X_boundary(1:(Mx+1)/2)=linspace(-1,0,(Mx+1)/2).^2-0.5;
X_boundary((Mx+1)/2:Mx)=linspace(0,1,(Mx+1)/2).^2-0.5;
for i = 1:(Mx-3)/2
    Y_boundary(1+i)=subs(y,x,X_boundary(1+i));
end
for i = 1:(Mx-3)/2
    Y_boundary((Mx+1)/2+i)=-subs(y,x,X_boundary((Mx+1)/2+i));
end
%定义四个边界
for i=1:Mx
    X(1,i)=X_boundary(i);
    X(My,i)=15/2*cos(2*pi*(i-1)/(Mx-1));
    Y(1,i)=Y_boundary(i);
    Y(My,i)=15/2*sin(2*pi*(i-1)/(Mx-1));
end
% for i=1:My
%     X(i,1)=0.5+7*((i-1)/(My-1))^2;
%     X(i,Mx)=0.5+7*((i-1)/(My-1))^2;
%     Y(i,1)=0;
%     Y(i,Mx)=0;
% end
slope=1;
X(:,1)=sine_interpolation(0.5,7.5,My,slope);
X(:,Mx)=sine_interpolation(0.5,7.5,My,slope);
Y(:,1)=zeros(My,1);
Y(:,Mx)=zeros(My,1);
%plot(X(1,:),Y(1,:));
%%
%计算P,Q
% P1x = -([X(1,2:end),2*X(1,end)-X(1,end-1)]+[2*X(1,1)-X(1,2),X(1,1:end-1)]-2*X(1,:))./abs([X(1,2:end),2*X(1,end)-X(1,end-1)]-X(1,:))/2-lambda*([X(1,2:end),2*X(1,end)-X(1,end-1)]-X(1,:)).*(((X(3,1)+X(1,1)-2*X(2,1))/2)./((X(2,1)-X(1,1)).^2)+((X(3,end)+X(1,end)-2*X(2,end))/2)./((X(2,end)-X(1,end)).^2))/2;
% P3x = -([X(end,2:end),2*X(end,end)-X(end,end-1)]+[2*X(end,1)-X(end,2),X(end,1:end-1)]-2*X(end,:))./abs([X(end,2:end),2*X(end,end)-X(end,end-1)]-X(end,:))/2-lambda*([X(end,2:end),2*X(end,end)-X(end,end-1)]-X(end,:)).*(((X(end-2,1)+X(end,1)-2*X(end-1,1))/2)./((X(end-1,1)-X(end,1)).^2)+((X(end-2,end)+X(end,end)-2*X(end-1,end))/2)./((X(end-1,end)-X(end,end)).^2))/2;
% P1y = -([Y(1,2:end),2*Y(1,end)-Y(1,end-1)]+[2*Y(1,1)-Y(1,2),Y(1,1:end-1)]-2*Y(1,:))./abs([Y(1,2:end),2*Y(1,end)-Y(1,end-1)]-Y(1,:))/2-lambda*([Y(1,2:end),2*Y(1,end)-Y(1,end-1)]-Y(1,:)).*(((Y(3,1)+Y(1,1)-2*Y(2,1))/2)./((Y(2,1)-Y(1,1)).^2)+((Y(3,end)+Y(1,end)-2*Y(2,end))/2)./((Y(2,end)-Y(1,end)).^2))/2;
% P3y = -([Y(end,2:end),2*Y(end,end)-Y(end,end-1)]+[2*Y(end,1)-Y(end,2),Y(end,1:end-1)]-2*Y(end,:))./abs([Y(end,2:end),2*Y(end,end)-Y(end,end-1)]-Y(end,:))/2-lambda*([Y(end,2:end),2*Y(end,end)-Y(end,end-1)]-Y(end,:)).*(((Y(end-2,1)+Y(end,1)-2*Y(end-1,1))/2)./((Y(end-1,1)-Y(end,1)).^2)+((Y(end-2,end)+Y(end,end)-2*Y(end-1,end))/2)./((Y(end-1,end)-Y(end,end)).^2))/2;
% Px = zeros(My,Mx); Py = zeros(My,Mx);
% %test!!!
% P1x = 0.1*cot(2*pi*linspace(0,1,Mx)+0.01);
% P3x = 0.1*cot(2*pi*linspace(0,1,Mx)+0.01);
% for i = 1:My
%     Px(i,:)=P1x+(P3x-P1x)*(i-1)/(My-1);
%     Py(i,:)=P1y+(P3y-P1y)*(i-1)/(My-1);
% end
% X = X';Y = Y';%记得改回来!!!!
% Q4x = -([X(1,2:end),2*X(1,end)-X(1,end-1)]+[2*X(1,1)-X(1,2),X(1,1:end-1)]-2*X(1,:))./abs([X(1,2:end),2*X(1,end)-X(1,end-1)]-X(1,:))/2-lambda*([X(1,2:end),2*X(1,end)-X(1,end-1)]-X(1,:)).*(((X(3,1)+X(1,1)-2*X(2,1))/2)./((X(2,1)-X(1,1)).^2)+((X(3,end)+X(1,end)-2*X(2,end))/2)./((X(2,end)-X(1,end)).^2))/2;
% Q2x = -([X(end,2:end),2*X(end,end)-X(end,end-1)]+[2*X(end,1)-X(end,2),X(end,1:end-1)]-2*X(end,:))./abs([X(end,2:end),2*X(end,end)-X(end,end-1)]-X(end,:))/2-lambda*([X(end,2:end),2*X(end,end)-X(end,end-1)]-X(end,:)).*(((X(end-2,1)+X(end,1)-2*X(end-1,1))/2)./((X(end-1,1)-X(end,1)).^2)+((X(end-2,end)+X(end,end)-2*X(end-1,end))/2)./((X(end-1,end)-X(end,end)).^2))/2;
% Q4y = -([Y(1,2:end),2*Y(1,end)-Y(1,end-1)]+[2*Y(1,1)-Y(1,2),Y(1,1:end-1)]-2*Y(1,:))./abs([Y(1,2:end),2*Y(1,end)-Y(1,end-1)]-Y(1,:))/2-lambda*([Y(1,2:end),2*Y(1,end)-Y(1,end-1)]-Y(1,:)).*(((Y(3,1)+Y(1,1)-2*Y(2,1))/2)./((Y(2,1)-Y(1,1)).^2)+((Y(3,end)+Y(1,end)-2*Y(2,end))/2)./((Y(2,end)-Y(1,end)).^2))/2;
% Q2y = -([Y(end,2:end),2*Y(end,end)-Y(end,end-1)]+[2*Y(end,1)-Y(end,2),Y(end,1:end-1)]-2*Y(end,:))./abs([Y(end,2:end),2*Y(end,end)-Y(end,end-1)]-Y(end,:))/2-lambda*([Y(end,2:end),2*Y(end,end)-Y(end,end-1)]-Y(end,:)).*(((Y(end-2,1)+Y(end,1)-2*Y(end-1,1))/2)./((Y(end-1,1)-Y(end,1)).^2)+((Y(end-2,end)+Y(end,end)-2*Y(end-1,end))/2)./((Y(end-1,end)-Y(end,end)).^2))/2;
% Qx = zeros(My,Mx); Qy = zeros(My,Mx);
% for i = 1:Mx
%     Qx(:,i)=Q4x+(Q4x-Q2x)*(i-1)/(Mx-1);
%     Qy(:,i)=Q4y+(Q4y-Q2y)*(i-1)/(Mx-1);
% end
% X =X';Y=Y';%这里改回来了
P1=-ones(1,Mx);
P3=-ones(1,Mx);
Q2=-ones(My,1)./linspace(0.00001,1,My)'/100;
Q4=-ones(My,1)./linspace(0.00001,1,My)'/100;
P = zeros(My,Mx);Q = zeros(My,Mx);
% for i = 1:My
%     P(i,:)=P1+(P3-P1)*(i-1)/(My-1);
% end
% for i = 1:Mx
%     Q(:,i)=Q4+(Q4-Q2)*(i-1)/(Mx-1);
% end
%%
%初始化，超限插值
%X(2:My-1,2:Mx-1)=rand(My-2,Mx-2);
%Y(2:My-1,2:Mx-1)=rand(My-2,Mx-2);
for i = 1:My
    for j = 1:Mx
        xq_array = linear_interpolation(X(i,1),X(i,Mx),Mx);
        xr_array = sine_interpolation(X(1,j),X(My,j),My,slope);
        temp_1 = sine_interpolation(X(1,1),X(My,1),My,slope);
        temp_2 = sine_interpolation(X(1,Mx),X(My,Mx),My,slope);
        xs_array = linear_interpolation(temp_1(i),temp_2(i),Mx);
        X(i,j) = xq_array(j)+xr_array(i)-xs_array(j);
        yq_array = linear_interpolation(Y(i,1),Y(i,Mx),Mx);
        yr_array = sine_interpolation(Y(1,j),Y(My,j),My,slope);
        temp_1 = sine_interpolation(Y(1,1),Y(My,1),My,slope);
        temp_2 = sine_interpolation(Y(1,Mx),Y(My,Mx),My,slope);
        ys_array = linear_interpolation(temp_1(i),temp_2(i),Mx);
        Y(i,j) = yq_array(j)+yr_array(i)-ys_array(j);
    end
end
%展示初始网格
figure(1)
plot(X,Y,'-k');
hold on;
plot(X',Y','-k');
set(gca,'position',[0,0,1,1])
xlim([-1.5,1.5]);
ylim([-1.5,1.5]);
%%
%定义X(i,j+1),X(i,j-1)等等
[X0p,Xp0,X0m,Xm0,Xpp,Xpm,Xmp,Xmm]=deal(X);
[Y0p,Yp0,Y0m,Ym0,Ypp,Ypm,Ymp,Ymm]=deal(Y);
step=0;
while true
    X0p(1:My-1,:)=X(2:My,:);X0p(My,:)=2*X(My,:)-X(My-1,:);
    X0m(2:My,:)=X(1:My-1,:);X0m(1,:)=2*X(1,:)-X(2,:);
    Xp0(:,1:Mx-1)=X(:,2:Mx);Xp0(:,Mx)=X(:,2);
    Xm0(:,2:Mx)=X(:,1:Mx-1);Xm0(:,1)=X(:,Mx-1);
    Y0p(1:My-1,:)=Y(2:My,:);Y0p(My,:)=2*Y(My,:)-Y(My-1,:);
    Y0m(2:My,:)=Y(1:My-1,:);Y0m(1,:)=2*Y(1,:)-Y(2,:);
    Yp0(:,1:Mx-1)=Y(:,2:Mx);Yp0(:,Mx)=Y(:,2);
    Ym0(:,2:Mx)=Y(:,1:Mx-1);Ym0(:,1)=Y(:,Mx-1);
    
    alpha=(0.5*(X0p-X0m)).^2+(0.5*(Y0p-Y0m)).^2;
    beta=0.25*(X0p-X).*(Xp0-X)+0.25*(Y0p-Y).*(Yp0-Ym0);
    gama=(0.5*(Xp0-Xm0)).^2+(0.5*(Yp0-Ym0)).^2;%中心差分
    J=(X0p-X).*(Yp0-Y)-(Xp0-X).*(Y0p-Y);
       
    Xpp(:,1:Mx-1)=X0p(:,2:Mx); Xpp(:,Mx)=X0p(:,2);
    Xpm(:,1:Mx-1)=X0m(:,2:Mx); Xpm(:,Mx)=X0m(:,2);
    Xmp(1:My-1,:)=Xm0(2:My,:); Xmp(My,:)=2*Xm0(My,:)-Xm0(My-1,:);
    Xmm(2:My,:)=Xm0(1:My-1,:); Xmm(1,:)=2*Xm0(1,:)-Xm0(2,:);
    Ypp(:,1:Mx-1)=Y0p(:,2:Mx); Ypp(:,Mx)=Y0p(:,2);
    Ypm(:,1:Mx-1)=Y0m(:,2:Mx); Ypm(:,Mx)=Y0m(:,2);
    Ymp(1:My-1,:)=Ym0(2:My,:); Ymp(My,:)=2*Ym0(My,:)-Ym0(My-1,:);
    Ymm(2:My,:)=Ym0(1:My-1,:); Ymm(1,:)=2*Ym0(1,:)-Ym0(2,:);
    
    bw=alpha;be=alpha;bs=gama;bn=gama;bp=bw+be+bs+bn;
    cpx=-beta.*(0.5*(Xpp-Xpm-Xmp+Xmm));
    cpy=-beta.*(0.5*(Ypp-Ypm-Ymp+Ymm));
    
    %Px=zeros(My,Mx);Qx=zeros(My,Mx);Py=zeros(My,Mx);Qy=zeros(My,Mx);
    NewX=(bw.*Xm0+be.*Xp0+bs.*X0m+bn.*X0p+cpx+(J.^2).*(P.*(Xp0-X)+Q.*(X0p-X)))./bp;
    NewY=(bw.*Ym0+be.*Yp0+bs.*Y0m+bn.*Y0p+cpy+(J.^2).*(P.*(Yp0-Y)+Q.*(Y0p-Y)))./bp;

    delta_x=max(max(abs(NewX(2:My-1,2:Mx-1)-X(2:My-1,2:Mx-1))));
    delta_y=max(max(abs(NewY(2:My-1,2:Mx-1)-Y(2:My-1,2:Mx-1))));
    X(2:My-1,1:Mx)=NewX(2:My-1,1:Mx);
    Y(2:My-1,2:Mx-1)=NewY(2:My-1,2:Mx-1);
    
    if max(delta_x,delta_y)<tol
        break
    end
    step = step+1;
    if step > 100
        break
    end
end
figure(2)
plot(X,Y,'-k');
hold on;
plot(X',Y','-k');
set(gca,'position',[0,0,1,1])
xlim([-1.5,1.5]);
ylim([-1.5,1.5]);
% figure(2)
% plot(X(:,51),'-k','linewidth',2);
% xlabel('结点编号')
% ylabel('X坐标')
function out_array = linear_interpolation(Y1,YM,Mx)
% define linear interpolation function as the blending function
    out_array = ones(Mx,1);
    for i = 1:Mx
        out_array(i,1) = Y1+(YM-Y1)*(i-1)/(Mx-1);
    end
end
function out_array = sine_interpolation(X1,XM,My,B_slope)
% define Sine interpolation function as the blending function
    out_array = ones(My,1);
    for i = 1:My
        out_array(i,1) = X1+(XM-X1)*sinh(B_slope*(i-1)/(My-1))/sinh(B_slope);
    end
end