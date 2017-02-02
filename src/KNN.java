import java.io.*;
import java.util.*;
import java.text.DecimalFormat;

public class KNN{

	private ArrayList<ArrayList<Double>> dataset;
	private ArrayList<ArrayList<Double>> columns;	// needed to normalise values & calculate weightings
	//private static double [] euclideanWeighting;
	private static DecimalFormat df; // limits to 4 decimal places
	private String path;        //API input value
	private Boolean headings;//API input value
	private int numColumns;//API input value
	private ArrayList<Integer> columnsToUse;//API input value
	private int predictionColumn;//API input value

	public KNN(String path, Boolean headings, int numColumns, int predictionColumn) {
		this.path = path;
		this.headings = headings;
		this.numColumns = numColumns; //convert number to index
		this.predictionColumn = predictionColumn;
		dataset = new ArrayList<>();
		columns = new ArrayList<>();
		df = new DecimalFormat("#.####"); // limits decimals to 4 decimal places
		for (int i = 0; i < numColumns; i++ ) { // this just instantiates the blank arraylists in columns so that readDataset works
			ArrayList<Double> x = new ArrayList<>();
			columns.add(x);
		}
	}

	private void readDataset() {
		String currentRow = "";
		try (BufferedReader br = new BufferedReader(new FileReader(path))) {
			if(headings){
				String headings = br.readLine(); // this takes the column headings out if they are there
			}
			while ((currentRow = br.readLine()) != null) {
				ArrayList<Double> row = toDoubleArrayList(currentRow.split(",")); // splits string into string [] then into ArrayList<double>
				dataset.add(row);
				// add to columns
				for (int i = 0; i < numColumns; i++) {
					columns.get(i).add(row.get(i)); // e.g this will add the school attribute to the arrilist of all the school values for each row
				}
			}
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static ArrayList<Double> toDoubleArrayList(String [] values) {
		ArrayList<Double> result = new ArrayList<>();
		for (String a: values) {
			result.add(Double.parseDouble(a));
		}
		return result;
	}
	
	private void trimDataset(){
		//read idexes of columns to use
		columnsToUse = new ArrayList<Integer>();
		System.out.println("Enter indexs of columns to be used for prediction in range 0 - " + numColumns);
		Scanner s = new Scanner(System.in);
        Scanner numScanner = new Scanner(s.nextLine()); // read indexes in one line
        while (numScanner.hasNextInt()) {
            columnsToUse.add(numScanner.nextInt());
        } //columnsToUse now contains the indexes of columns to be used, predictionColumn holds one to predict
		
		ArrayList<ArrayList<Double>> newColumns = new ArrayList<ArrayList<Double>>();
		for(int a: columnsToUse){
			newColumns.add(columns.get(a));
		} // adds only the columns needed for prediction to newColumns
		newColumns.add(columns.get(predictionColumn)); //add predictionColumn last
		
		ArrayList<ArrayList<Double>> newDataset = new ArrayList<ArrayList<Double>>();
		int columnSize = newColumns.get(0).size();
		
		for(int i = 0; i < columnSize; i++){ //for each row
			ArrayList<Double> currentRow = new ArrayList<Double>();
			for(int j = 0; j < newColumns.size(); j++){ //add each column value
				currentRow.add(newColumns.get(j).get(i));
			}
			newDataset.add(currentRow);
		}
		dataset = newDataset;
		columns = newColumns;		
	}
	
	private void normaliseDataset() { // all values in the range 0-1 to stop skewing by attributes such as age
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
			for (int i = 0; i < size; i++) { // loop through  rows to be normalised
				double oldValue = current.get(i);
				double normValue = Double.parseDouble(df.format((oldValue - minValues[i])/(maxValues[i]- minValues[i]))); // normalisation formula
				current.set(i, normValue);
			}
		}
		/*
		for(ArrayList<Double> a : dataset){
			System.out.print("[");
			for(Double x : a){
				System.out.print(x + " ");
			}
			System.out.println("]");
		}
		*/
	}
	
	public static void main(String [] args) {
		KNN z = new KNN("student-mat-normalised.csv", true, 33, 32);
		z.readDataset();
		z.trimDataset();
		z.normaliseDataset();
	}
	
}
















