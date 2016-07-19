/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lregression;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author Ashish
 */
public class LRegression {

    String fPath;
    List<String> vocabList = new ArrayList<>();	
    boolean sWord = false,testSet = false;

    public void setIsSWord(boolean sWord) {
        this.sWord = sWord;
    }
    Set<String> stopList = new HashSet<>();
    ArrayList<HashMap <String, Integer>> weightArray = new ArrayList<HashMap <String, Integer>>(1000);
    int ind = 0,ham = 0,spam=0;;
    HashMap tList = new HashMap();
    HashMap wList = new HashMap();
    HashMap testWtList = new HashMap();
    HashMap prList = new HashMap();
    HashMap dList = new HashMap();
    
    public static void main(String[] args) throws IOException  {
        LRegression l = new LRegression();
        l.readFilesforTraining(args[0]);
        l.readFilesforTraining(args[1]);
        l.trainLR();
        l.testLR(args[2],args[3]);
        System.out.println("With Stopword excluded:");
        LRegression l2 = new LRegression();
        l2.setIsSWord(true);
        l2.findStopWords();
        l2.readFilesforTraining(args[0]);
        l2.readFilesforTraining(args[1]);
        l2.trainLR();
        l2.testLR(args[2],args[3]);
    }
    
    void readFilesforTraining(String folderPath) throws IOException{
        fPath = folderPath;
        File f = new File(folderPath);//"C:\\Users\\Ashish\\Documents\\ML2\\train\\ham"
//        File f = new File("C:\\Users\\Ashish\\Documents\\ML2\\train\\ham");//"C:\\Users\\Ashish\\Documents\\ML2\\train\\ham"
        FilenameFilter textFilter;
        textFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.toLowerCase().endsWith(".txt");
            }
        };

        File[] files = f.listFiles(textFilter);
        for (File file : files) {
//            System.out.println(file.getPath());
            extractWordsFromFile(file.getPath());
        }

    }

    private void extractWordsFromFile(String path) throws FileNotFoundException, IOException {
        String csvSplitBy = " ";
        for(String line : Files.readAllLines(Paths.get(path) , StandardCharsets.ISO_8859_1)){		
            for(String s : line.split(csvSplitBy)){
                Pattern p = Pattern.compile("[^A-Za-z0-9]");
                Matcher m = p.matcher(s);
                boolean b = m.find();
                if (b == false){
                    if(fPath.contains("train")){
                        vocabList.add(s);
                        if(sWord)
                            if(stopList.contains(s))
                                vocabList.remove(s);
                        
                    }
                    else {
                        vocabList.add(s);
                        if(sWord)
                            if(stopList.contains(s))
                                vocabList.remove(s);
                    }
                       
                }
            }
      }
        Iterator i = vocabList.listIterator();
        HashMap<String,Integer> myMap1 = new HashMap<String, Integer>();
        while(i.hasNext()){
            String k = (String) i.next();
            if(myMap1.containsKey(k))
                myMap1.replace(k, myMap1.get(k)+1);
            else
                myMap1.put(k, 1);
            if(!testSet)
                wList.put(k, 0.1);        //initialize all weights to 1
        }
//        System.out.println(wList);
        weightArray.ensureCapacity(1000);
        weightArray.add(ind, myMap1);
        prList.put(ind, 0.5);      //initialize all Pr[i] to 0.5
        ind++;
//        System.out.println(myMap1);
        tList.putAll(myMap1);
        
//        myMap1.clear();
        vocabList.clear();
        
    }
    
    public void trainLR(){
        dList.putAll(wList);
        dList.replaceAll((k,v) -> 0);
//        System.out.println(weightArray);
        for(int i = 0;i<100;i++){
            calculateProbMap(i);
            calculatedWMap();
            calculateWeightMap(i);
        }
//        System.out.println(wList);
    }

    public void findStopWords() throws IOException{
        String csvFile = "C:\\Users\\Ashish\\Documents\\ML2\\stopword.txt";
	
        BufferedReader br = null;
	String line = "";
	String csvSplitBy = " ";
        int j=0;

	try {

		br = new BufferedReader(new FileReader(csvFile));
		while ((line = br.readLine()) != null) {

		        // use comma as separator
                    String[] trainingStr = line.split(csvSplitBy);
                    stopList.addAll(Arrays.asList(trainingStr));        //note this step. Good method to convert into arraylist
                    
                }
                
	} catch (FileNotFoundException e) {
	} catch (IOException e) {
	} finally {
		if (br != null) {
			try {
				br.close();
			} catch (IOException e) {
                        }
		}
	}	
    }

    private void calculateProbMap(int i) {
        int index = 0;
        while(index!=ind){
            double theta = summation(index,i);
            double prob = sigmoid(theta);
            prList.replace(index, prob);
            index++;
        }
        
    }

    private double summation(int index,int count) {
        double sum = 0,n;
        Set s = wList.entrySet();
        Iterator i = s.iterator();
        HashMap<String, Integer> tmpData = (HashMap<String, Integer>) weightArray.get(index);
        while(i.hasNext()){
            Map.Entry m = (Map.Entry) i.next();
            String k = (String) m.getKey();
            if(tmpData.containsKey(k))    
                n=tmpData.get(k);
            else 
                n = 0;
            double p = 0;
            if(count==0)
                p = (double)m.getValue()*n;
            else
                p = (double)m.getValue()*n;
            sum+=p;
        }
        return sum+1;
    }

    private double sigmoid(double theta) {
        if(theta>=100)
            return 1;
        if(theta<-100)
            return 0;
        theta*=-1;
        double num = Math.exp(theta);
        return num/(1.0+num);
    }

    private void calculatedWMap() {
        //initialize dList
        dList.replaceAll((k,v)->0);
        //now calculate dw
        Set s = wList.entrySet();
        
        Iterator it = s.iterator();
        while(it.hasNext()){
            Map.Entry m = (Map.Entry) it.next();
            String k = (String) m.getKey();
            
                
            int index = 0;
            double dw = 0.0;
            while(index!=ind){
                HashMap<String, Integer> tmpData = (HashMap<String, Integer>) weightArray.get(index);
                int dataji = 0,datajn = 1;
                if(tmpData.containsKey(k)){
                    dataji = tmpData.get(k);
                }
                if(index>=340)
                    datajn = 0;
                double prj = (double) prList.get(index);
                if (index == 0){
                    dw = dw + (double)(dataji*(datajn-prj));
                }
                else
                    dw = dw + (double)(dataji*(datajn-prj));
                
                
                index++;
            }
            dList.replace(k, dw);
        }
//        System.out.println(dList);
    }

    private void calculateWeightMap(int count) {
        double neta = 0.08, lambda = 0.003;
        double wi = 0.0;
        Set s = wList.entrySet();
        Iterator it = s.iterator();
        while(it.hasNext()){
            Map.Entry m = (Map.Entry) it.next();
            String k = (String) m.getKey();
            if(count == 0)
                wi = (double)m.getValue() + (double)neta*((double)dList.get(k) - lambda*(double)m.getValue());
            else
                wi = (double)m.getValue() + (double)neta*((double)dList.get(k) - lambda*(double)m.getValue());
            wList.replace(k, wi);
        }
//        System.out.println(wList);
    }
    
    public void testLR(String hamTest, String spamTest) throws IOException{
        testSet = true;
        ind = 0;
        weightArray.clear();
        prList.clear();
        readFilesforTrainingTest(hamTest);
        double acc = (ham/348d) *100;
        int sp = spam,h = ham;
//        System.out.println("Ham="+ham+"\nSpam="+spam+" Ham Accuracy="+acc);
        
        readFilesforTrainingTest(spamTest);
        acc = (spam - sp)/130d *100;
//        System.out.println("Ham="+ham+"\nSpam="+spam+" Spam accuracy="+acc);
        acc = (h+spam-sp)/478d * 100;
        System.out.println("Overall accuracy="+acc);
        
        weightArray.clear();
        
    }

    private void readFilesforTrainingTest(String folderPath) throws IOException {
        fPath = folderPath;
        File f = new File(folderPath);//"C:\\Users\\Ashish\\Documents\\ML2\\train\\ham"
//        File f = new File("C:\\Users\\Ashish\\Documents\\ML2\\train\\ham");//"C:\\Users\\Ashish\\Documents\\ML2\\train\\ham"
        FilenameFilter textFilter;
        textFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.toLowerCase().endsWith(".txt");
            }
        };

        File[] files = f.listFiles(textFilter);
        for (File file : files) {
//            System.out.println(file.getPath());
            extractWordsFromFileTest(file.getPath());
        }
    }

    private void extractWordsFromFileTest(String path) throws IOException {
        String csvSplitBy = " ";
        for(String line : Files.readAllLines(Paths.get(path) , StandardCharsets.ISO_8859_1)){		
            for(String s : line.split(csvSplitBy)){
                Pattern p = Pattern.compile("[^A-Za-z0-9]");
                Matcher m = p.matcher(s);
                boolean b = m.find();
                if (b == false){
                    if(fPath.contains("train")){
                        vocabList.add(s);
                        if(sWord)
                            if(stopList.contains(s))
                                vocabList.remove(s);
                        
                    }
                    else {
                        vocabList.add(s);
                        if(sWord)
                            if(stopList.contains(s))
                                vocabList.remove(s);
                    }
                       
                }
            }
      }
        Iterator i = vocabList.listIterator();
        HashMap<String,Integer> myMap1 = new HashMap<String, Integer>();
        while(i.hasNext()){
            String k = (String) i.next();
            if(myMap1.containsKey(k))
                myMap1.replace(k, myMap1.get(k)+1);
            else
                myMap1.put(k, 1);
            
        }
//        System.out.println(wList);
        ind++;
        testWtList.putAll(myMap1);
        //create separate probability calculatoin method for this
        calculateTestProb();
        testWtList.clear();
        vocabList.clear();
    }

    private void calculateTestProb() {
        Set s = wList.entrySet();
        double sum = 0.0,w;
        int x = 0;
        Iterator i = s.iterator();
        while(i.hasNext()){
            Map.Entry m = (Map.Entry) i.next();
            if(testWtList.containsKey(m.getKey()))
                x = (int)testWtList.get(m.getKey());
            w = (double)m.getValue();
            sum+=w*x;
        }
        double sig = sigmoid(sum+1);//TO DO:: create new sigmoid using log. Prob values are underflowing.
        if(sig>=0.5)
            ham++;
        else
            spam++;
    }
    
}
