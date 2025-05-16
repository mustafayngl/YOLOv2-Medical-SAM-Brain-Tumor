function yolo_test_with_visuals()
    % Load test table and trained detector
    load('tumorYolo2Detector.mat', 'detector');   % assumes detector variable exists
    load('yolo_testTbl.mat', 'testTbl');          % loads testTbl variable

    % Create datastore from testTbl
    imds = imageDatastore(testTbl.imageFilename);
    blds = boxLabelDatastore(testTbl(:, 'tumor'));
    testDS = combine(imds, blds);
    testDS = transform(testDS, @convertTo3Channel);

    % Detect objects on the test set
    detectionResults = detect(detector, testDS);

    % Evaluate results
    [ap, recall, precision] = evaluateDetectionPrecision(detectionResults, testTbl(:, 'tumor'));

    % Show metrics
    mAP = mean(ap);
    meanPrecision = mean(precision, 'omitnan');
    meanRecall = mean(recall, 'omitnan');
    f1 = 2 * (meanPrecision * meanRecall) / (meanPrecision + meanRecall);

    fprintf('Test Results:\n');
    fprintf('Mean Average Precision (mAP): %.3f\n', mAP);
    fprintf('Precision: %.3f\n', meanPrecision);
    fprintf('Recall: %.3f\n', meanRecall);
    fprintf('F1-score: %.3f\n\n', f1);

    % Visualize precision-recall
    figure;
    if iscell(recall)
        plot(recall{1}, precision{1}, 'LineWidth', 2);
    else
        plot(recall, precision, 'LineWidth', 2);
    end
    xlabel('Recall'); ylabel('Precision');
    title(sprintf('PR Curve (mAP = %.3f, F1 = %.3f)', mAP, f1));
    grid on;

    % Optionally visualize a few detections
    for i = 1:5
        I = imread(testTbl.imageFilename{i});
        I = convertTo3Channel({I});
        [bboxes, scores, labels] = detect(detector, I{1}, 'Threshold', 0.5);
        detected = insertObjectAnnotation(I{1}, 'Rectangle', bboxes, scores);
        groundTruth = insertShape(I{1}, 'Rectangle', testTbl.tumor{i}, 'Color', 'green', 'LineWidth', 2);

        figure, 
        subplot(1,2,1), imshow(detected), title('YOLOv2 Detection');
        subplot(1,2,2), imshow(groundTruth), title('Ground Truth');
    end
end

% Helper to convert grayscale images to RGB
function dataOut = convertTo3Channel(data)
    img = data{1};
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    data{1} = img;
    dataOut = data;
end
