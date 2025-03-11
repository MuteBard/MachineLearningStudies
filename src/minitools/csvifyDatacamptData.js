function csvify(rawTableStr){
    let updatedStr = (" "+rawTableStr).replace(/(?![\r\n])\s+/g, ",");
    const csvList = updatedStr.split('\n');
    const updatedCsvList = csvList.map((row) => {
        firstCommaIndex = row.indexOf(',')
        const updatedRow = row.substring(firstCommaIndex + 1);
        return updatedRow;
    })
    return updatedCsvList.join('\n')
}


const data = `   Team League  Year   RS   RA   W    G  Playoffs
0   TEX     AL  2012  808  707  93  162         1
1   TEX     AL  2011  855  677  96  162         1
2   TEX     AL  2010  787  687  90  162         1
3   TEX     AL  2009  784  740  87  162         0
4   TEX     AL  2008  901  967  79  162         0
5   TEX     AL  2007  816  844  75  162         0
6   TEX     AL  2006  835  784  80  162         0
7   TEX     AL  2005  865  858  79  162         0
8   TEX     AL  2004  860  794  89  162         0
9   TEX     AL  2003  826  969  71  162         0
10  TEX     AL  2002  843  882  72  162         0
11  TEX     AL  2001  890  968  73  162         0
12  TEX     AL  2000  848  974  71  162         0
13  TEX     AL  1999  945  859  95  162         1
14  TEX     AL  1998  940  871  88  162         1
15  TEX     AL  1997  807  823  77  162         0
16  TEX     AL  1996  928  799  90  163         1
17  TEX     AL  1993  835  751  86  162         0
18  TEX     AL  1992  682  753  77  162         0
19  TEX     AL  1991  829  814  85  162         0
20  TEX     AL  1990  676  696  83  162         0
21  TEX     AL  1989  695  714  83  162         0
22  TEX     AL  1988  637  735  70  161         0
23  TEX     AL  1987  823  849  75  162         0
24  TEX     AL  1986  771  743  87  162         0
25  TEX     AL  1985  617  785  62  161         0
26  TEX     AL  1984  656  714  69  161         0
27  TEX     AL  1983  639  609  77  163         0
28  TEX     AL  1982  590  749  64  162         0
29  TEX     AL  1980  756  752  76  163         0
30  TEX     AL  1979  750  698  83  162         0
31  TEX     AL  1978  692  632  87  162         0
32  TEX     AL  1977  767  657  94  162         0
33  TEX     AL  1976  616  652  76  162         0
34  TEX     AL  1975  714  733  79  162         0
35  TEX     AL  1974  690  698  83  161         0
36  TEX     AL  1973  619  844  57  162         0`

console.log(csvify(data))
// On terminal execute:  
// node.exe csvifyDatacamptData > rangers.csv