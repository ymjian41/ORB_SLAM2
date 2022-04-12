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
#include<unistd.h>
#include<chrono>
#include<iomanip>
#include <Eigen/Dense>
#include <math.h>

#include<opencv2/core/core.hpp>

#include"System.h"

using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

vector<string> readLine(string filename);
vector<float> splitString(string s);
void get_data(vector<string> poses, vector<vector<float>> &DeepVO_output);
Eigen::Matrix<float, 3, 3> Rotation(float thetax, float thetay, float thetaz);

int main(int argc, char **argv)
{
    vector<string> poses = readLine("/home/ymjian/DeepVO-pytorch/result/out_09.txt");
    // std::cout<<"here1"<<std::endl;
    vector<vector<float>> DeepVO_output;
    get_data(poses, DeepVO_output);
    Eigen::Matrix<float, 3, 3> R;
    Eigen::Matrix<float, 3, 1> t;
    Eigen::Matrix4f T_now = Eigen::Matrix4f::Zero();
    Eigen::Matrix<float, 4, 4> T_Last;
    Eigen::Matrix<float, 4, 4> T_rel;
    // std::cout<<"here2"<<std::endl;
    // vector<Eigen::Matrix<double, 4, 4>> deepvo_T;
    vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>> deepvo_T;
    for(int i = 0; i < DeepVO_output.size(); i++){
        if(i == 0)
        {
            // std::cout << "here3" << std::endl;
            R = Rotation(DeepVO_output[i][1], DeepVO_output[i][0], DeepVO_output[i][2]);
            t << DeepVO_output[i][3],
                 DeepVO_output[i][4],
                 DeepVO_output[i][5];
            T_now.block(0,0,3,3) = R;
            T_now.block(0,3,3,1) = t;
            T_now(3,3) = 1.0;
            // cout << "R = " << endl << " "  << T_now << endl << endl;
            deepvo_T.push_back(T_now);
            // std::cout << "pushback0" << std::endl;
        }
        else
        {
            // std::cout << "else" << std::endl;
            T_Last = T_now;
            // std::cout << "else1" << std::endl;
            R = Rotation(DeepVO_output[i][1], DeepVO_output[i][0], DeepVO_output[i][2]);
            t << DeepVO_output[i][3],
                 DeepVO_output[i][4],
                 DeepVO_output[i][5];
            // std::cout << "else2" << std::endl;
            T_now.block(0,0,3,3) = R;
            T_now.block(0,3,3,1) = t / 20;
            T_now(3,3) = 1.0;
            // std::cout << "before mul" << std::endl;
            T_rel = T_now * (T_Last.inverse()); // left
            // T_rel = (T_Last.inverse()) * T_now;
            // std::cout << "multiply\n" << T_rel.dtype() << std::endl;
            // cout<<"Rel "<<T_rel<<endl;
            // auto TT = T_rel.data();
            deepvo_T.push_back(T_rel);
            cout << "Trel\n" << T_rel << endl;
            // std::cout << "pushback" << std::endl;
        }

    }
    std::cout<< "deepvo size: " << deepvo_T.size()<<endl;

    // cout<<"Matrix = "<<endl<<deepvo_T[0]<<endl;

    // }

    std::cout<<"finish: " << deepvo_T[0] << std::endl;
    // for (int i = 0; i < deepvo_T.size(); i++)
    // {
    //     cout << deepvo_T[i] << endl;
    // }
    // cout << 'deep\n' <<  deepvo_T << endl;

    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe,deepvo_T);

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
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");    

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
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

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}

vector<string> readLine(string filename){
	vector<string> valid_commands;
	ifstream word_file(filename);
	if(word_file.is_open()){
		string word;
		while(getline(word_file, word)){
			valid_commands.push_back(word);
		}
		word_file.close();
	}
	return valid_commands;
}

vector<float> splitString(string s){

	std::string delimiter = ", ";
	size_t pos = 0;
	std::string token;
	vector<float> v;

	pos = s.find(delimiter);
	while((pos = s.find(delimiter)) != std::string::npos){
		token = s.substr(0, pos);
		float num = stof(token);
		v.push_back(num);
		s.erase(0, pos + delimiter.length());
	}
	float num = stof(s);
	v.push_back(num);
	return v;
}

void get_data(vector<string> poses, vector<vector<float>> &DeepVO_output)
{
    for (auto pose : poses){
            
			DeepVO_output.push_back(splitString(pose));

		}
} 

Eigen::Matrix<float, 3, 3> Rotation(float thetax, float thetay, float thetaz){

    Eigen::Matrix3f Rx = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f Ry = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f Rz = Eigen::Matrix3f::Zero();

    float r11, r12, r13, r21, r22, r23, r31, r32, r33;
    r11 = r12 = r13 = r21 = r22 = r23 = r31 = r32 = r33 = 0.0;
    Rx(0,0) = 1.0;
    Rx(1,1) = cos(thetax);
    Rx(1,2) = -1.0 * sin(thetax);
    Rx(2,1) = sin(thetax);
    Rx(2,2) = cos(thetax);

    
    Ry(0,0) = cos(thetay);
    Ry(0,2) = sin(thetay);
    Ry(1,1) = 1.0;
    Ry(2,1) = -1.0 * sin(thetay);
    Ry(2,2) = cos(thetay);

    Rz(0,0) = cos(thetaz);
    Rz(0,1) = -1.0 * sin(thetaz);
    Rz(1,0) = sin(thetaz);
    Rz(1,1) = cos(thetaz);
    Rz(2,2) = 1.0;
    
    // cout << "Rx = " << endl << " "  << Rx << endl << endl;
    // cout << "Ry = " << endl << " "  << Ry << endl << endl;
    // cout << "Rz = " << endl << " "  << Rz << endl << endl;

    Eigen::Matrix<float, 3, 3> R = Ry * (Rx * Rz);
    return R;

}
