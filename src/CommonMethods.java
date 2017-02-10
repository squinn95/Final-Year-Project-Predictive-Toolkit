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
		/*
		for(ArrayList<Double> a : dataset){ //for each row
			System.out.print("[");
			for(Double x : a){
				System.out.print(x + " ");
			}
			System.out.println("]");
		}
		*/
	}
	
	protected static ArrayList<ArrayList<Double>> classifyDataset(ArrayList<ArrayList<Double>> dataset, int [] splitPoints) { // this sets up classes in training set, adds new column, if needed for continuous sets
		if(splitPoints.length > 0){
			//ArrayList<ArrayList<Double>> newDataset = (ArrayList<ArrayList<Double>>)dataset.clone();
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
						current.add(new Double(classCount));
						assigned = true;
						break loop;
					}
					classCount++;
				}
				if(assigned == false){
					//class = classCount
					current.add(new Double(classCount));
					assigned = true;
				}
				//System.out.println(value + ": " + classCount );
			}
			return newDataset;
		}
		else{
			return dataset;
		}
		/*
		for(ArrayList<Double> a : dataset){ //for each row
			System.out.print("[");
			for(Double x : a){
				System.out.print(x + " ");
			}
			System.out.println("]");
		}
		*/
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
				trainingSet.add(iter5.next()); // 200 rows
			else
				validationSet.add(iter5.next());// 195
		}
		Pair<ArrayList<ArrayList<Double>>,ArrayList<ArrayList<Double>>> output = new Pair(trainingSet, validationSet);
		return output;
	}
	
}




































