clear

trial_len = 60;
n_samples = 750;
input_side_dim = 36;
data = zeros(n_samples, trial_len, input_side_dim, input_side_dim, 3);
hand = zeros(n_samples, trial_len, 2);

for sample=1:n_samples
    while true
        hand_pos = [randi([1, input_side_dim]), randi([1, input_side_dim])];    
        target_pos = [randi([1, input_side_dim]), randi([1, input_side_dim])];
        if target_pos(1)~=hand_pos(1) %&& norm(target_pos-hand_pos)>5
            break
        end
    end
    target_onset = randi([10, 20]);

    diff = target_pos-hand_pos;
    if hand_pos(1)>target_pos(1)
        x = target_pos(1):.01:hand_pos(1);
        x = fliplr(x);
    else
        x = hand_pos(1):.01:target_pos(1);
    end
    y = hand_pos(2)+(x-hand_pos(1))*diff(2)/diff(1);
    x = round(x); y = round(y);
    A = [x', y'];
    B = unique(A, 'stable', 'rows');
    j = 1;

    for i=1:trial_len
        red = zeros(input_side_dim,input_side_dim); green = red; blue = red;

        if i>=target_onset
            red(target_pos(1),target_pos(2)) = 255;
        end

        green(hand_pos(1),hand_pos(2)) = 255;
        rgbImage = cat(3,red,green,blue);
        data(sample,i,:,:,:) = rgbImage/255;
        hand(sample,i,:) = hand_pos/input_side_dim;
        
        if i>=target_onset+1 && (hand_pos(1)~=target_pos(1) || hand_pos(2)~=target_pos(2))
            j = j+1;
            hand_pos(1) = B(j,1);
            hand_pos(2) = B(j,2);
        end

%         imshow(rgbImage,'InitialMagnification','fit')
%         drawnow
%         pause(0.02)
    end
end

save('train750_36x36_len60.mat', 'data', 'hand')
