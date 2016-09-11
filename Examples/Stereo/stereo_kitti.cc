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

#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

#define MATCHING_PAIR_THRESHOLD 5

using namespace std;

extern void (*logKeys)(std::vector<cv::KeyPoint> &, long unsigned int);
extern void (*logKFs)(ORB_SLAM2::KeyFrame *);
extern void (*logMapPts)(ORB_SLAM2::MapPoint *);
extern void (*logLoopObs)(ORB_SLAM2::KeyFrame *, ORB_SLAM2::KeyFrame *, std::vector<ORB_SLAM2::MapPoint*> &, cv::Mat &);
extern bool (*isInstLoop)(ORB_SLAM2::LoopClosing *);
extern bool (*insLoopObs)(ORB_SLAM2::LoopClosing *);

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

// hook function for logging all map points
void testlogLoopObs(ORB_SLAM2::KeyFrame *pCurrentKF, ORB_SLAM2::KeyFrame *pMatchedKF, std::vector<ORB_SLAM2::MapPoint*> &vMatchedPts, cv::Mat &mScw);
ofstream fLogLoopObs;

// hook function for inserting loop detection
bool testIsInsLoop(ORB_SLAM2::LoopClosing *pLoopClosing);
char szLoopObsFileName[32] = {0};

// hook function for inserting loop detection
bool testInsLoopObs(ORB_SLAM2::LoopClosing *pLoopClosing);

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

    // init log file for loop observations
    logLoopObs = testlogLoopObs;
    fLogLoopObs.open("/tmp/logLoopObs.txt", ios_base::out);

    isInstLoop = testIsInsLoop;
    insLoopObs = testInsLoopObs;

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

void testlogLoopObs(ORB_SLAM2::KeyFrame *pCurrentKF, ORB_SLAM2::KeyFrame *pMatchedKF, std::vector<ORB_SLAM2::MapPoint*> &vMatchedPts, cv::Mat &mScw)
{
    int matchedSize = vMatchedPts.size();
    int rows = mScw.rows;
    int cols = mScw.cols;
    std::vector<ORB_SLAM2::MapPoint*> vMatched;
    std::vector<int> vIdx;

    for (int i = 0; i < matchedSize; i++) {
        if (vMatchedPts[i]) {
            vMatched.push_back(vMatchedPts[i]);
            vIdx.push_back(i);
        }
    }
    matchedSize = vMatched.size();

    fLogLoopObs << pCurrentKF->mnFrameId << " " << pCurrentKF->mnId << " " << pMatchedKF->mnFrameId << " " << pMatchedKF->mnId << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fLogLoopObs << mScw.at<float>(i*cols+j) << " ";
        }
        fLogLoopObs << std::endl;
    }
    fLogLoopObs << matchedSize << std::endl;
    for (int i = 0; i < matchedSize; i++) {
        int idxM = vMatched[i]->GetIndexInKeyFrame(pMatchedKF);
        fLogLoopObs << "    " << pCurrentKF->mvKeysUn[vIdx[i]].pt.x << " " << pCurrentKF->mvKeysUn[vIdx[i]].pt.y << " to " << pMatchedKF->mvKeysUn[idxM].pt.x << " " << pMatchedKF->mvKeysUn[idxM].pt.y << std::endl;
    }

}

bool testIsInsLoop(ORB_SLAM2::LoopClosing *pLoopClosing)
{
    struct stat buffer;
    sprintf(szLoopObsFileName, "/tmp/%lu.txt", pLoopClosing->mpCurrentKF->mnFrameId);
    return (stat (szLoopObsFileName, &buffer) == 0);
}

bool testInsLoopObs(ORB_SLAM2::LoopClosing *pLoopClosing)
{
    ORB_SLAM2::KeyFrame* pCurrentKF = pLoopClosing->mpCurrentKF;
    ORB_SLAM2::KeyFrame* pMatchedKF = NULL;
    unsigned long currentFrameId = pCurrentKF->mnFrameId;
    unsigned long matchedFrameId = 0;
    int matchedSize = 0, nInitialCandidates = 0, nInliers = 0;
    vector<ORB_SLAM2::MapPoint*> vpMapPoints1, vpMapPointMatches;
    ORB_SLAM2::ORBmatcher matcher(0.75,true);
    ORB_SLAM2::Sim3Solver *pSolver = NULL;
    cv::Mat R, t;
    float s;
    g2o::Sim3 gScm, gSmw;
    bool found = false;
    ifstream fGetLoopObs;

    if (!testIsInsLoop(pLoopClosing)) return false;

    fGetLoopObs.open(szLoopObsFileName);

    {
        stringstream ss;
        string s;

        getline(fGetLoopObs,s);
        ss << s;
        ss >> matchedFrameId;
        ss >> matchedSize;
    }

    nInitialCandidates = pLoopClosing->mvpEnoughConsistentCandidates.size();

    for(int i=0; i<nInitialCandidates; i++)
    {
        ORB_SLAM2::KeyFrame* pKF = pLoopClosing->mvpEnoughConsistentCandidates[i];
        if (pKF->mnFrameId == matchedFrameId) {
            pMatchedKF = pKF;
            break;
        }
    }

    if (!pMatchedKF) {
        printf("The matched frame[%lu] of current frame[%lu] is not in the candidates with enough consistent.\n",
               matchedFrameId, currentFrameId);
        goto END;
    }

    vpMapPoints1 = pCurrentKF->GetMapPointMatches();
    vpMapPointMatches = vector<ORB_SLAM2::MapPoint*>(vpMapPoints1.size(),static_cast<ORB_SLAM2::MapPoint*>(NULL));

    for (int i = 0; i < matchedSize; i++) {
        stringstream ss;
        string s;
        cv::KeyPoint p1, p2;
        int idxCPF = -1, idxMPF = -1;
        int nKey1 = (int)pCurrentKF->mvKeys.size();
        int nKey2 = (int)pMatchedKF->mvKeys.size();

        getline(fGetLoopObs,s);
        ss << s;
        ss >> p1.pt.x;
        ss >> p1.pt.y;
        ss >> p2.pt.x;
        ss >> p2.pt.y;

        for (int j = 0; j < nKey1; j++) {
            if (1 > cv::norm(pCurrentKF->mvKeys[j].pt-p1.pt)) {
                idxCPF = j;
                break;
            }
        }

        for (int j = 0; j < nKey2; j++) {
            if (1 > cv::norm(pMatchedKF->mvKeys[j].pt-p2.pt)) {
                idxMPF = j;
                break;
            }
        }

        if (0 > idxCPF || 0 > idxMPF) {
            continue;
        }

        vpMapPointMatches[idxCPF] = pMatchedKF->GetMapPoint(idxMPF);
        nInliers++;
    }

    pSolver = new ORB_SLAM2::Sim3Solver(pCurrentKF,pMatchedKF,vpMapPointMatches,pLoopClosing->mbFixScale);
    pSolver->SetRansacParameters(0.99,20,300);

    R = pSolver->GetEstimatedRotation();
    t = pSolver->GetEstimatedTranslation();
    s = pSolver->GetEstimatedScale();
    matcher.SearchBySim3(pCurrentKF,pMatchedKF,vpMapPointMatches,s,R,t,7.5);

    gScm = g2o::Sim3(ORB_SLAM2::Converter::toMatrix3d(R),ORB_SLAM2::Converter::toVector3d(t),s);
    gSmw = g2o::Sim3(ORB_SLAM2::Converter::toMatrix3d(pMatchedKF->GetRotation()),ORB_SLAM2::Converter::toVector3d(pMatchedKF->GetTranslation()),1.0);
    pLoopClosing->mg2oScw = gScm*gSmw;
    pLoopClosing->mScw = ORB_SLAM2::Converter::toCvMat(pLoopClosing->mg2oScw);

    pLoopClosing->mvpCurrentMatchedPoints = vpMapPointMatches;

    if (MATCHING_PAIR_THRESHOLD > nInliers) {
        printf("The matching pair[%d] is less than threshold[%d]\n",
               nInliers, MATCHING_PAIR_THRESHOLD);
        goto END;
    }

    found = true;

END:
    fGetLoopObs.close();
    return found;
}
