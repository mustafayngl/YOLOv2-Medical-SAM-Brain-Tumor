function test_yolov2_with_medsam()
    %% 1. Load trained YOLOv2 detector
    load('tumorYolo2Detector.mat', 'detector');
    fprintf('Detector network class: %s\n', class(detector.Network));

    %% 2. Load test data
    load('yolo_testTbl.mat', 'testTbl');

    %% 3. Load Medical SAM
    medsam = medicalSegmentAnythingModel;

    %% 4. SAM Input Size
    samInputSize = [256 256];

    %% 5. Test only 5 images (or fewer if table is smaller)
    numTests = min(5, height(testTbl));

    for i = 1:numTests
        % Read and prepare image
        I = imread(testTbl.imageFilename{i});
        I_rgb = ensureRGB(I);
        originalSize = size(I_rgb);
        I_resized = imresize(I_rgb, samInputSize);

        % YOLOv2 detection
        [bboxes, scores] = detect(detector, I_rgb, 'Threshold', 0.3);

        if isempty(bboxes)
            fprintf('❌ No detections in: %s\n', testTbl.imageFilename{i});
            continue;
        end

        % Extract embeddings
        embeddings = extractEmbeddings(medsam, I_resized);

        % Segment for each box
        for j = 1:size(bboxes, 1)
            box = bboxes(j, :);
            scaleX = samInputSize(2) / originalSize(2);
            scaleY = samInputSize(1) / originalSize(1);
            boxPrompt = [box(1)*scaleX, box(2)*scaleY, box(3)*scaleX, box(4)*scaleY];

            mask = segmentObjectsFromEmbeddings(medsam, embeddings, samInputSize, BoundingBox=boxPrompt);
            maskResized = imresize(mask, originalSize(1:2));

            % Visualization
            figure;
            imshow(I_rgb);
            hold on;
            rectangle('Position', box, 'EdgeColor', 'cyan', 'LineWidth', 2);
            visboundaries(maskResized, 'Color', 'r', 'LineWidth', 1);
            title(sprintf('Detection %d: %s', i, testTbl.imageFilename{i}), 'Interpreter', 'none');
            hold off;
        end
    end
end

%% Helper
function Iout = ensureRGB(I)
    if size(I,3) == 1
        Iout = repmat(I, [1 1 3]);
    else
        Iout = I;
    end
end
