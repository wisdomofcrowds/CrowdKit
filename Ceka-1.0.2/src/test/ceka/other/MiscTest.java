/**
 * 
 */
package ceka.other;

import java.util.ArrayList;

import ceka.utils.Misc;

/**
 * @author Zhang
 *
 */
public class MiscTest {
	
public static void main(String[] args) {
		
	try {
		
		for (int i = 1; i <= 10; i++)
		{
			ArrayList<Integer> list = Misc.randSelect(i, 1, 10);
			for (int j = 0; j < list.size(); j++) {
				System.out.print(list.get(j) + " ");
			}
			System.out.print("\n");
		}
		
	} catch (Exception e) {
		
		e.printStackTrace();
	}
}		

}
