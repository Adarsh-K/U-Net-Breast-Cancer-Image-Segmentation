%%%% This simple script creates Canny Edge "overlayed" image for an input image
%%%% I've used this to create "overlayed" images for Training and Test/validation Image datasets 
%%%% Note: The segmentation labels are NOT to be "overlayed" with their corresponding Canny edges

clc;clear all;
for i=1:17 % Number of images to convert, 17 images in 'test' directory
    A=imread(sprintf('/Users/adarshkumar/Desktop/data1/test/images/img/%d.png',i-1));
    E = edge(rgb2gray(A),'canny'); % Get edge
    B = imoverlay(A,E); % Overlays "Canny Edges" got above on image A we read 
    imwrite(B, sprintf('/Users/adarshkumar/Desktop/canny/test/images/img/%d.png',i-1));
end