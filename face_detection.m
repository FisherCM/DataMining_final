person = zeros(112 * 92,40);

tail = '.pgm';
%先将每一个人的7个训练样本取均值后再作为PCA算法的训练样本输入
for num = 1:40
    folder = strcat('s',num2str(num),'\');
    for i = 1:7
        path = strcat(folder,num2str(i),tail);
        img = double(imread(path));
        person(:,num) = person(:,num) + reshape(img,112 * 92,1);
    end

    person(:,num) = person(:,num) ./ 7;
    s_mean = reshape(person(:,num),112,92);
    s_mean = uint8(s_mean);
    write_path = strcat('person_mean',num2str(num),tail);
    imwrite(s_mean,write_path);
end

mean = zeros(112 * 92,1);
for i = 1:40
    mean = mean + person(:,i);
end
mean = mean ./ 40;

c_mat = zeros(112 * 92,40);
for i = 1:40
    c_mat(:,i) = person(:,i) - mean;
end

[e_vec,e_val] = eig(c_mat * c_mat');
val = diag(e_val);
[sort_val,index] = sort(val,'descend');

%-------------------------------------------------------------------
%此处指定所用的k值
k = 30;
%-------------------------------------------------------------------
eig_mat = zeros(112 * 92,k);
for i = 1:k
    eig_mat(:,i) = e_vec(:,index(i));
end

projection = zeros(k,40);
for i = 1:40
    projection(:,i) = eig_mat' * (person(:,i) - mean);
end

correction = 0;
for num = 1:40
    folder = strcat('s',num2str(num),'\');
    for i = 8:10
        path = strcat(folder,num2str(i),tail);
        test_img = double(imread(path));
        test_vec = reshape(test_img,112 * 92,1);
        test_val = eig_mat' * (test_vec - mean);
        distance = zeros(40,1);

        for id = 1:40
            distance(id,1) = norm(projection(:,id) - test_val);
        end

        [sort_dis,position] = sort(distance);
        result = position(1);
        
        if result == num
            correction = correction + 1;
        end
    end
end

%得到准确率
correct_rate = correction / 120;
correct_rate

%具体显示某一个测试样本和与之匹配的训练样本
test_img = double(imread('s1\8.pgm'));
test_vec = reshape(test_img,112 * 92,1);
test_val = eig_mat' * (test_vec - mean);
distance = zeros(40,1);

for id = 1:40
    distance(id,1) = norm(projection(:,id) - test_val);
end

[sort_dis,position] = sort(distance);
result = position(1);

sample = reshape(person(:,result),112,92);
subplot(2,1,1);
imshow(uint8(test_img));
subplot(2,1,2);
imshow(uint8(sample));


