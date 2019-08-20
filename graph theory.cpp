floyd
/*********************************/
void floyd() {
	ff(i,1,n)
	ff(j,1,n)
		if(i==j) e[i][j]=0;
			else e[i][j]=INF;

	ff(i,1,n)
	ff(j,1,n) {
		char c;
		cin >> c;
		if(c=='1') e[i][j]=1;
	}

    ff(k,1,n)
    	ff(i,1,n)
    		ff(j,1,n)
                if(e[i][j]>e[i][k]+e[k][j] ) 
                    e[i][j]=e[i][k]+e[k][j];
}
/*********************************/
