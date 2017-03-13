//package prediction;
import java.io.*;
import java.util.*;
import java.text.DecimalFormat;
import java.util.concurrent.ThreadLocalRandom;

class Pair<A,B> {
    private A a;
    private B b;

    public Pair(A a, B b) {
        this.a = a;
        this.b = b;
    }
	
	public A getFirst(){
		return a;
	}
	
	public B getSecond(){
		return b;
	}
};

public class CommonMethods{
	protected static ArrayList<ArrayList<Double>> readDatasetFile(String filePath, boolean headings) {
		ArrayList<ArrayList<Double>> dataset = new ArrayList<ArrayList<Double>>();
		String currentRow = "";
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))){
			if(headings){
				String removeHeadings = br.readLine(); // this takes the column headings out if they are there
			}
			while ((currentRow = br.readLine()) != null) {
				ArrayList<Double> row = toDoubleArrayList(currentRow.split(",")); // splits string into string [] then into ArrayList<double>
				dataset.add(row);
			}
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
		return dataset;
	}
	
	protected static ArrayList<ArrayList<Double>> createColumnStructure(ArrayList<ArrayList<Double>> rowInput){
		//for each row, add element at each index to arraylist at that index in columns
		ArrayList<ArrayList<Double>> columns = new ArrayList<ArrayList<Double>>();
		for(int i = 0; i < rowInput.get(0).size(); i++){ //for each column
			ArrayList<Double> current = new ArrayList<Double>();
			for(ArrayList<Double> row: rowInput){
				current.add(row.get(i));
			}
			columns.add(current);
		}
		return columns;
	}
		
	protected static ArrayList<ArrayList<Double>> createRowStructure(ArrayList<ArrayList<Double>> columnInput){
		//for each row, add element at each index to arraylist at that index in columns
		ArrayList<ArrayList<Double>> rows = new ArrayList<ArrayList<Double>>();
		for(int i =0; i < columnInput.get(0).size(); i++ ){ //for each row
			ArrayList<Double> current = new ArrayList<Double>();
			for (ArrayList<Double> column: columnInput) {
				current.add(column.get(i)); 
			}
			rows.add(current);
		}
		return rows;
	}
	
	protected static ArrayList<ArrayList<Double>> trimColumns(ArrayList<ArrayList<Double>> columns, int [] columnsToUse, int predictionColumn){
		ArrayList<ArrayList<Double>> newColumns = new ArrayList<ArrayList<Double>>();
		for(int a: columnsToUse){
			newColumns.add(columns.get(a));
		} // adds only the columns needed for prediction to newColumns
		newColumns.add(columns.get(predictionColumn)); //add predictionColumn last
		return newColumns;
	}
	
	protected static ArrayList<ArrayList<Double>> normaliseDataset(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> columns) { // all values in the range 0-1 to stop skewing by attributes such as age
		DecimalFormat df = new DecimalFormat("#.####");
		ArrayList<ArrayList<Double>> newDataset = new ArrayList<ArrayList<Double>>();
		int size = columns.size() -1; //-1 because we dont want to normalise prediction column
		double [] maxValues = new double [size];
		double [] minValues = new double [size]; // these will hold max and min values for each column
		for (int i = 0; i < size; i++) {
			maxValues[i] = Collections.max(columns.get(i));
			minValues[i] = Collections.min(columns.get(i));
		}
		
		Iterator<ArrayList<Double>> iter = dataset.iterator();
		while (iter.hasNext()) { // loop through rows
			ArrayList<Double> current = iter.next(); // current row
			ArrayList<Double> newRow = new ArrayList<Double>();
			for (int i = 0; i < size; i++) { // loop through  rows to be normalised
				double oldValue = current.get(i);
				double normValue = Double.parseDouble(df.format((oldValue - minValues[i])/(maxValues[i]- minValues[i]))); // normalisation formula
				newRow.add(normValue);
			}
			newRow.add(current.get(current.size()-1)); //add last element the prediction column
			newDataset.add(newRow);
		}
		return newDataset;
	}
	
	protected static ArrayList<ArrayList<Double>> classifyDataset(ArrayList<ArrayList<Double>> dataset, int [] splitPoints) { // this sets up classes in training set, replacing continuous value with class
		if((splitPoints != null) && (splitPoints.length > 0)){
			ArrayList<ArrayList<Double>> newDataset = new ArrayList<ArrayList<Double>>(dataset);
			Iterator<ArrayList<Double>> iter4 = newDataset.iterator();
			while (iter4.hasNext()) { // loop through rows
				ArrayList<Double> current = iter4.next(); // current row
				Double value = current.get(current.size() -1); //last row is predictionColumn since trim
				int classCount = 1;
				boolean assigned = false;
				loop:
				for(int i = 0; i < splitPoints.length; i++){
					if(value < splitPoints[i]){
						//class = classCount
						current.remove(current.size() -1); //replace last element with class
						current.add(new Double(classCount));
						assigned = true;
						break loop;
					}
					classCount++;
				}
				if(assigned == false){
					//class = classCount
					current.remove(current.size() -1); 
					current.add(new Double(classCount));
					assigned = true;
				}
				//System.out.println(value + ": " + classCount );
				//System.out.println(current);
			}
			return newDataset;
		}
		else{
			return dataset;
		}
	}
	
	protected static ArrayList<Double> toDoubleArrayList(String [] values) {
		ArrayList<Double> result = new ArrayList<>();
		for (String a: values) {
			result.add(Double.parseDouble(a));
		}
		return result;
	}
	
	protected static Pair<ArrayList<ArrayList<Double>>,ArrayList<ArrayList<Double>>> randomlyHalfSet(ArrayList<ArrayList<Double>> inputSet){ //splits it roughly in half
		ArrayList<ArrayList<Double>> trainingSet = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> validationSet = new ArrayList<ArrayList<Double>>();
		Iterator<ArrayList<Double>> iter5 = inputSet.iterator(); // splits into two sets
        int rnd;
		while (iter5.hasNext()) { // loop through rows
			rnd = ThreadLocalRandom.current().nextInt(1, 11);
			if (rnd <= 5)
				trainingSet.add(iter5.next()); 
			else
				validationSet.add(iter5.next());
		}
		Pair<ArrayList<ArrayList<Double>>,ArrayList<ArrayList<Double>>> output = new Pair(trainingSet, validationSet);
		return output;
	}
	
	static boolean equalsDouble(Double a, Double b){
		return (Double.doubleToLongBits(a) == Double.doubleToLongBits(b));
	}
	
	static int getColumnCount(ArrayList<Double> list, Double value){
		int count = 0;
		for(Double d: list){
				if(CommonMethods.equalsDouble(d,value)){
					count++;
				}
		}
		return count;
	}
	
	static HashMap<Double,ArrayList<ArrayList<Double>>> getIndexColumnStructuresMap(ArrayList<ArrayList<Double>> trainSet, HashMap<Double,ArrayList<Integer>> indexLocationMap){
		HashMap<Double,ArrayList<ArrayList<Double>>> indexColumnStructures = new HashMap<Double,ArrayList<ArrayList<Double>>>();
		for(Double value: indexLocationMap.keySet()){ //for each value associated with this attribute
			ArrayList<ArrayList<Double>> currentValueRowStructure = new ArrayList<ArrayList<Double>>();
			for(Integer index: indexLocationMap.get(value)){ //for each row number associated with this value
				currentValueRowStructure.add(trainSet.get(index));
			}
			ArrayList<ArrayList<Double>> currentValueColumnStructure = CommonMethods.createColumnStructure(currentValueRowStructure); //convert to column format
			indexColumnStructures.put(value,currentValueColumnStructure); //put class, column structure in result
		}
		return indexColumnStructures;
	}
	
	static HashMap<Double,ArrayList<Integer>> getColumnValueLocationMap(ArrayList<ArrayList<Double>> trainColumns, int index){
		ArrayList<Double> targetColumn = trainColumns.get(index); //last column
		
		HashMap<Double,ArrayList<Integer>> indexMap = new HashMap<Double,ArrayList<Integer>>();
		//ArrayList<Double> result = new ArrayList<Double>();
		
		for(int i=0; i < targetColumn.size(); i++){
			Double value = targetColumn.get(i);
			if(indexMap.containsKey(value)){
					ArrayList<Integer> current = indexMap.get(value);
					current.add(i); //add this index to the list of indexes corresponding to this class value
					indexMap.put(value, current);
			}
			else{
				ArrayList<Integer> current = new ArrayList<Integer>();
				current.add(i);
				indexMap.put(value, current);
			}
		}
		
		return indexMap;
	}
	
	static Double getMaxCount(HashMap<Double,Double> classCounts){
		//create tree map with comparitor which sorts by value instead of key
		TreeSet<Map.Entry<Double, Double>> entriesSet = new TreeSet<>(new Comparator<Map.Entry<Double, Double>>(){
           @Override 
			public int compare(Map.Entry<Double, Double> x, Map.Entry<Double, Double> y) {
				return x.getValue().compareTo(y.getValue());
			}
        });
        entriesSet.addAll(classCounts.entrySet());
		return entriesSet.last().getKey();
	}
	
	static Double getMinCount(HashMap<Double,Double> classCounts){
		//create tree map with comparitor which sorts by value instead of key
		TreeSet<Map.Entry<Double, Double>> entriesSet = new TreeSet<>(new Comparator<Map.Entry<Double, Double>>(){
           @Override 
			public int compare(Map.Entry<Double, Double> x, Map.Entry<Double, Double> y) {
				return x.getValue().compareTo(y.getValue());
			}
        });
        entriesSet.addAll(classCounts.entrySet());
		return entriesSet.first().getKey();
	}
	
	static double euclideanDistance(ArrayList<Double> a, ArrayList<Double> b){ 
		DecimalFormat df = new DecimalFormat("#.####");
		double total = 0;
		for(int i = 0; i < a.size() - 1; i++){
			double difference = (a.get(i) - b.get(i));
			total += (difference * difference); //replace 1 with weighting scheme
		}
		return Double.parseDouble(df.format(Math.sqrt(total)));
	}
	
	static double manhattanDistance(ArrayList<Double> a, ArrayList<Double> b){ 
		DecimalFormat df = new DecimalFormat("#.####");
		double total = 0;
		for(int i = 0; i < a.size() - 1; i++){
			double difference = (a.get(i) - b.get(i));
			total += Math.abs(difference); 
		}
		return total;
	}
	
	static double calcDistance(ArrayList<Double> a, ArrayList<Double> b, String formula){
			if(formula.equals("manhattan"))
				return manhattanDistance(a, b);
			else
				return euclideanDistance(a, b);
	}
	
}




































