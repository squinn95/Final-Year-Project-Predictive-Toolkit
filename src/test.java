import java.io.*;
import java.util.*;

public class test{
	public static void main(String [] args){
		ArrayList<Integer> nums = new ArrayList<Integer>();
		System.out.println("Enter indexs of columns to be used for prediction in range 0 - \"numvalues\"");
		Scanner s = new Scanner(System.in);
        Scanner numScanner = new Scanner(s.nextLine()); // read indexes in one line
        while (numScanner.hasNextInt()) {
            nums.add(numScanner.nextInt());
        } 	
        for(int a: nums)
			System.out.println(a);
	}
}