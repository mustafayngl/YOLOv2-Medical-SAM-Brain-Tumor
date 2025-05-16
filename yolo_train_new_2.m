function yolo_train
    %% 1. Load and Split Dataset
    load('YOLOv2_dataset.mat', 'T');
    T = T(randperm(height(T)), :);

    n = height(T);
    idx1 = round(0.7 * n);
    idx2 = round(0.85 * n);

    trainTbl = T(1:idx1, :);
    valTbl   = T(idx1+1:idx2, :);
    testTbl  = T(idx2+1:end, :);

    %% 2. Input Setup
    inputSize = [224 224 3];
    numClasses = 1;
    classes = {'tumor'};
    numAnchors = 5;

    %% 3. Estimate Anchor Boxes from training data
    % Scale bounding boxes to match inputSize
    scaledBBoxes = cell(height(trainTbl), 1);
    for i = 1:height(trainTbl)
        info = imfinfo(trainTbl.imageFilename{i});
        sx = inputSize(2) / info.Width;
        sy = inputSize(1) / info.Height;
        box = trainTbl.tumor{i};
        box(:, [1 3]) = box(:, [1 3]) * sx;
        box(:, [2 4]) = box(:, [2 4]) * sy;
        scaledBBoxes{i} = box;
    end

    tmpTbl = trainTbl(:, {'imageFilename'});
    tmpTbl.tumor = scaledBBoxes;

    imdsTmp = imageDatastore(tmpTbl.imageFilename);
    bldsTmp = boxLabelDatastore(tmpTbl(:, 'tumor'));
    dsTmp   = combine(imdsTmp, bldsTmp);
    dsTmp   = transform(dsTmp, @convertTo3Channel);

    anchors = estimateAnchorBoxes(dsTmp, numAnchors);
    fprintf('\n📦 Estimated Anchor Boxes [w h]:\n');
    disp(anchors);

    %% 4. Create Datastores with RGB fix
    trainDS = createYOLODatastore(trainTbl);
    valDS   = createYOLODatastore(valTbl);
    testDS  = createYOLODatastore(testTbl);

    %% 5. YOLOv2 Network (ResNet-50 backbone)
    net = resnet50();
    lgraph = yolov2Layers(inputSize, numClasses, anchors, net, 'activation_40_relu');

    %% 6. Training Options
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 1e-4, ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 16, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', valDS, ...
        'ValidationFrequency', floor(height(trainTbl)/16), ...
        'Verbose', true, ...
        'Plots', 'training-progress');

    %% 7. Train Detector
    detector = trainYOLOv2ObjectDetector(trainDS, lgraph, options);

    %% 8. Test Detection
    detectionResults = detect(detector, testDS);

    %% 9. Evaluate
    [ap, recall, precision] = evaluateDetectionPrecision(detectionResults, testTbl(:, 'tumor'));

    meanPrecision = mean(precision, 'omitnan');
    meanRecall = mean(recall, 'omitnan');
    f1Vec = 2 * (precision .* recall) ./ (precision + recall);
    f1 = max(f1Vec, [], 'omitnan');
    mAP = mean(ap);

    %% 10. Save Everything
    save('tumorYolo2Detector.mat', ...
        'detector', 'mAP', 'precision', 'recall', 'f1', 'anchors');

    %% 11. Report
    fprintf('\n✅ Training complete. Model saved to tumorYolo2Detector.mat\n');
    fprintf('Mean Average Precision (mAP): %.3f\n', mAP);
    fprintf('Precision: %.3f\n', meanPrecision);
    fprintf('Recall: %.3f\n', meanRecall);
    fprintf('F1-score: %.3f\n', f1);

    figure;
    plot(recall, precision, 'LineWidth', 2);
    xlabel('Recall'); ylabel('Precision');
    title(sprintf('PR Curve (mAP = %.3f, F1 = %.3f)', mAP, f1));
    grid on;
end

%% Helper: RGB Conversion Function
function dataOut = convertTo3Channel(data)
    I = data{1};
    if size(I, 3) == 1
        I = repmat(I, [1 1 3]);
    end
    data{1} = I;
    dataOut = data;
end

%% Helper: Create RGB-aware YOLOv2 Datastore
function ds = createYOLODatastore(tbl)
    imds = imageDatastore(tbl.imageFilename);
    blds = boxLabelDatastore(tbl(:, 'tumor'));
    ds   = combine(imds, blds);
    ds   = transform(ds, @convertTo3Channel);
end
