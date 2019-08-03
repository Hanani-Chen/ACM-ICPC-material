int getw(int year,int month,int day) {
	if(month<3){year-=1;month+=12;}
	int b[7]={7,1,2,3,4,5,6};
	int c=int(year/100),y=year-100*c;
	int w=int(c/4)-2*c+y+int(y/4)+(26*(month+1)/10)+day-1;
	w=(w%7+7)%7;
	return b[w];
}
