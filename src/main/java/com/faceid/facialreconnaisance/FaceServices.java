package com.faceid.facialreconnaisance;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;
import org.opencv.core.CvType;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfDMatch;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Size;
import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

public class ImageComparator {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the two images
        Mat image1 = Imgcodecs.imread("path/to/first/image.jpg");
        Mat image2 = Imgcodecs.imread("path/to/second/image.jpg");

        // Convert images to grayscale
        Mat grayImage1 = new Mat();
        Mat grayImage2 = new Mat();
        Imgproc.cvtColor(image1, grayImage1, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(image2, grayImage2, Imgproc.COLOR_BGR2GRAY);

        // Resize images to the same size for comparison (optional)
        Size newSize = new Size(800, 600);
        Imgproc.resize(grayImage1, grayImage1, newSize);
        Imgproc.resize(grayImage2, grayImage2, newSize);

        // Detect keypoints and extract descriptors
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
        detector.detect(grayImage1, keypoints1);
        detector.detect(grayImage2, keypoints2);

        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        extractor.compute(grayImage1, keypoints1, descriptors1);
        extractor.compute(grayImage2, keypoints2, descriptors2);

        // Match the descriptors
        MatOfDMatch matches = new MatOfDMatch();
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        matcher.match(descriptors1, descriptors2, matches);

        // Filter matches based on distance
        double maxDist = 0.4; // Maximum distance between descriptors for a match
        double minDist = 100.0; // Minimum distance for a good match

        List<DMatch> goodMatches = new ArrayList<>();
        for (DMatch match : matches.toArray()) {
            if (match.distance < maxDist) {
                goodMatches.add(match);
            }
        }

        // Draw the matches
        Mat outputImage = new Mat();
        MatOfByte drawnMatches = new MatOfByte();
        Features2d.drawMatches(grayImage1, keypoints1, grayImage2, keypoints2, new MatOfDMatch(goodMatches.toArray(new DMatch[0])), outputImage, Scalar.all(-1), Scalar.all(-1), drawnMatches, Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);

        // Calculate the similarity ratio
        double similarity = (double) goodMatches.size() / (double) keypoints1.size().height;

        // Display the similarity ratio
        System.out.println("Similarity: " + similarity);

        // Save the output image with matches
        Imgcodecs.imwrite("path/to/output/image.jpg", outputImage);
    }
}
