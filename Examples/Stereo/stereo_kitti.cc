/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

extern void (*logKeys)(std::vector<cv::KeyPoint> &, long unsigned int);
extern void (*logKFs)(ORB_SLAM2::KeyFrame *);
extern void (*logMapPts)(ORB_SLAM2::MapPoint *);

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

// hook function for logging keys from every frames
void testLogKeys(std::vector<cv::KeyPoint> &vKeys, long unsigned int id);
ofstream fLogKeys;

// hook function for logging keys from every key frames
void testLogKFs(ORB_SLAM2::KeyFrame *pKF);
ofstream fLogKFs;

// hook function for logging all map points
void testLogMapPts(ORB_SLAM2::MapPoint *pMapPt);
ofstream fLogMapPts;

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // init log file for keys from every frames
    logKeys = testLogKeys;
    fLogKeys.open("/tmp/logKeys.txt", ios_base::out);

    // init log file for keys from every key frames
    logKFs = testLogKFs;
    fLogKFs.open("/tmp/logKFs.txt", ios_base::out);

    // init log file for all map points
    logMapPts = testLogMapPts;
    fLogMapPts.open("/tmp/logMapPts.txt", ios_base::out);

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::STEREO,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imLeft, imRight;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft,imRight,tframe);        

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    fLogKeys.close();
    fLogKFs.close();
    fLogMapPts.close();

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

void testLogKeys(std::vector<cv::KeyPoint> &vKeys, long unsigned int id)
{
    int size = vKeys.size();

    fLogKeys << id << " " << size << endl;

    for (int i = 0; i < size; i++) {
        fLogKeys << "    " << vKeys[i].pt.x << " " << vKeys[i].pt.y << std::endl;
    }
}

void testLogKFs(ORB_SLAM2::KeyFrame *pKF)
{
    int size = pKF->N;
    const std::vector<cv::KeyPoint> &vKeys = pKF->mvKeys;

    fLogKFs << pKF->mnFrameId << " " << pKF->mnId << " " << size << endl;

    for (int i = 0; i < size; i++) {
        fLogKFs << "    " << vKeys[i].pt.x << " " << vKeys[i].pt.y << std::endl;
    }
}

void testLogMapPts(ORB_SLAM2::MapPoint *pMapPt)
{
    ORB_SLAM2::KeyFrame *pKF = pMapPt->GetReferenceKeyFrame();
    int id = pMapPt->GetIndexInKeyFrame(pKF);

    fLogMapPts << pMapPt->mnId << " " << pKF->mnFrameId << " " << pKF->mnId << " " << id << std::endl;
}
