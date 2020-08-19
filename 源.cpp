#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"
#include<iostream>
#include<io.h>
#include<string>
#include<vector>
#include<direct.h>    //_mkdir  
#include<io.h> //_access
#include<math.h>
#include "omp.h"
#include<algorithm>

#pragma comment(lib, "facedetection.lib")//引入链接库

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000

using namespace cv;
using namespace std;


void getFiles(const std::string & path, std::vector<std::string> & files);
int facedetectdemo(string file, string dirName);
int generatedataset(std::string path = ".\\path\\dataset");
int imagemean(Mat &tmp);
bool cmp(pair<int, string> &a, pair<int, string>&b);
int generateface(string facepath, std::string path = ".\\path\\dataset");

/*
由数量众多照片拼贴而成的马赛克图片,分2步：
1.generatedataset(path);

2.generateface(facepath,path);

*/
int main() {
	std::string path = ".\\path\\dataset";//多图像数据集

	//采用facedetection生成人脸数据集，存放在文件夹 .\\path\\dataset\\mygirl 下（自动新建）
	//generatedataset(path);


	string facepath = "C:\\Users\\Leon\\Desktop\\aa.jpg";//待合成的人脸图像
	generateface(facepath,path);//合并生成人脸图像
	return 0;
}

void getFiles(const std::string & path, std::vector<std::string> & files)
{
	//文件句柄
	long long hFile = 0;
	//文件信息，_finddata_t需要io.h头文件
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表

			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				char drive[_MAX_DRIVE] = { 0 };
				char dir[_MAX_DIR] = { 0 };
				char fname[_MAX_FNAME] = { 0 };
				char ext[_MAX_EXT] = { 0 };
				_splitpath_s(fileinfo.name, drive, dir, fname, ext);
				if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".JPG") == 0 || strcmp(ext, ".bmp") == 0) {
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				}

			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

int facedetectdemo(string file, string dirName) {

	char drive[_MAX_DRIVE] = { 0 };
	char dir[_MAX_DIR] = { 0 };
	char fname[_MAX_FNAME] = { 0 };
	char ext[_MAX_EXT] = { 0 };
	_splitpath_s(file.c_str(), drive, dir, fname, ext);
	//cout << drive << endl;
	//cout << dir << endl;
	//cout << fname << endl;
	string renamefile = dirName + string(fname);

	Mat image = imread(file);
	if (image.empty()) {
		fprintf(stderr, "Can not load the image file %s.\n", file);
		return -1;
	}
	//resize(image, image, Size(0, 0), 0.3, 0.3);
	resize(image, image, Size(2048, 2048 * float(image.rows) / float(image.cols)));

	//Mat result_cnn = image.clone();
	int * pResults = NULL;
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer) {
		fprintf(stderr, "Can not alloc buffer.\n");
		return -1;
	}
	pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);
	//printf("%d faces detected.\n", (pResults ? *pResults : 0));

	//print the detection results
	float walpha = 2.0;
	float halpha = 2.0;
	for (int i = 0; i < (pResults ? *pResults : 0); i++) {
		short * p = ((short*)(pResults + 1)) + 142 * i;
		int x = p[0] + (1 - walpha)*p[2] / 2;
		int y = p[1] + (1 - halpha)*p[3] / 2;
		int w = walpha*p[3];
		int h = halpha*p[3];
		int confidence = p[4];
		int angle = p[5];
		if (x < 0 || y<0 || x + w>image.cols || w<0 || y + h>image.rows || h < 0) {
			continue;
		}
		if (confidence > 95) {
			//cout << image.cols << " " << image.rows << endl;
			printf("face_rect=[%d, %d, %d, %d], confidence=%d, angle=%d\n", x, y, w, h, confidence, angle);
			//rectangle(result_cnn, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
			string tmp = renamefile + "_" + to_string(i) + ".bmp";
			cout << tmp << endl;
			Mat ROI = image(Rect(x, y, w, h)).clone();
			//imshow("ROI", ROI);
			//waitKey(0);
			imwrite(tmp, ROI);
		}
	}
	//imshow("result_cnn", result_cnn);
	//waitKey();
	free(pBuffer);

	return 0;
}

int generatedataset(std::string path) {


	std::vector<std::string> files;
	getFiles(path, files);
	//for (int i = 0; i < files.size(); i++) {
	//	cout << files[i] << endl;
	//}
	cout << "files.size(): " << files.size() << endl;
	string dirName = path + "\\" + "mygirl" + "\\";
	if (_access(dirName.c_str(), 0) == -1)
	{
		int i = _mkdir(dirName.c_str());
	}
	omp_set_num_threads(16);
#pragma omp parallel for
	for (int i = 0; i < files.size(); i++) {
		//cout << files[i] << endl;
		facedetectdemo(files[i], dirName);
	}

	return 0;
}

int imagemean(Mat &tmp) {
	double mr = 0;
	double mg = 0;
	double mb = 0;
	for (int i = 0; i < tmp.rows; i++) {
		for (int j = 0; j < tmp.cols; j++) {
			mr += tmp.at<Vec3b>(i, j)[0];
			mg += tmp.at<Vec3b>(i, j)[1];
			mb += tmp.at<Vec3b>(i, j)[2];
		}
	}
	mr /= tmp.rows*tmp.cols;
	mg /= tmp.rows*tmp.cols;
	mb /= tmp.rows*tmp.cols;
	return mr + (256 + mg) + (512 + mb);
}

bool cmp(pair<int, string> &a, pair<int, string>&b) {
	return a.first < b.first;
}

int generateface(string facepath, std::string path) {
	Mat src = imread(facepath);

	resize(src, src, Size(64, 64*float(src.rows)/float(src.cols)));
	cout << src.size() << endl;
	imshow("src",src);
	waitKey(0);
	int rows = src.rows;
	int cols = src.cols;
	int base = 128;
	Mat newIm = Mat(rows * base, cols * base, CV_8UC3);

	path += "\\mygirl";
	std::vector<std::string> files;
	getFiles(path, files);

	vector<pair<int,string>> means;
	for (int i = 0; i < files.size(); i++) {
		Mat tmp = imread(files[i]);
		means.push_back(make_pair(imagemean(tmp), files[i]));
		//cout << means[i].first << "\t" << means[i].second << endl;
	}
	
	sort(means.begin(), means.end(), cmp);
	for (int i = 0; i < means.size(); i++) {
		cout << means[i].first << "\t" << means[i].second << endl;
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int m = src.at<Vec3b>(i, j)[0] + 256 + src.at<Vec3b>(i, j)[1] + 512 + src.at<Vec3b>(i, j)[2];
			string tmpstr = means[means.size()-1].second;
			for (int k = 0; k < means.size() - 1; k++) {
				if (means[k].first<=m && m<means[k + 1].first) {
					tmpstr = means[k].second;
					//cout << means[k].first << "\t" << tmpstr << endl;
					break;
				}
			}
			Mat tmp = imread(tmpstr);
			resize(tmp, tmp, Size(base, base));
			for (int xx = 0; xx < base; xx++) {
				for (int yy = 0; yy < base; yy++) {
					newIm.at<Vec3b>(yy + i * base, xx + j * base)[0] = tmp.at<Vec3b>(yy, xx)[0];
					newIm.at<Vec3b>(yy + i * base, xx + j * base)[1] = tmp.at<Vec3b>(yy, xx)[1];
					newIm.at<Vec3b>(yy + i * base, xx + j * base)[2] = tmp.at<Vec3b>(yy, xx)[2];
				}
			}
		}
	}

	char drive[_MAX_DRIVE] = { 0 };
	char dir[_MAX_DIR] = { 0 };
	char fname[_MAX_FNAME] = { 0 };
	char ext[_MAX_EXT] = { 0 };
	_splitpath_s(facepath.c_str(), drive, dir, fname, ext);
	cout << string(drive) + string(dir) + string(fname) + "_2.bmp" << endl;
	imwrite(string(drive)+string(dir)+string(fname)+"_2.bmp", newIm);

	
	return 0;
}
