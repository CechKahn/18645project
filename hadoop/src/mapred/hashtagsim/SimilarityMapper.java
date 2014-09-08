package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

/* SimilarityMapper -- second time mapper to map the fomat of fearture #hashtag1:count1;#hashtag2:count2;
 *	into hash_pair(e.g #hashtag1 #hahtag2 score). We first consider to use paire stirpe technuqie to further
 *	optimize the algorithm, but the hadoop's heap error bother me, so we revert back to a simple verion
 */
public class SimilarityMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		int i, j , score;
	
		String line = value.toString();
		//data's format is: feature #hashtag1:count1;#hahstag2:count2;
		String[] hashtag_featureVector = line.split("\\s+", 2); 

		String hashtags = hashtag_featureVector[1];

		//get each hash tag's key value pair
		String[] hashtag = hashtags.split(";");


		int length = hashtag.length;

		//split hashtag(length == 1, only one hashtag, no pair)
		if (length > 1) {

			String[] keys = new String[length];
			int[] values = new int[length];

			for (i = 0; i < length; i++) {
				String[] key_value = hashtag[i].split(":");
				keys[i] = key_value[0];
				values[i] = Integer.parseInt(key_value[1]);
			}
			StringBuilder builder = new StringBuilder();
			
			//get all hash pair
			for (i = 0; i < length - 1; i ++) {
				for (j = i + 1; j < length; j ++) {
		
					builder.setLength(0);
					score = values[i] * values[j];

					if ((keys[i]).compareTo(keys[j]) > 0) {
						builder.append(keys[j]);
						builder.append("\t");
						builder.append(keys[i]);
					} else {
						builder.append(keys[i]);
						builder.append("\t");
						builder.append(keys[j]);
					}
					//output hashpair and its score
					String hashtag_pair = builder.toString();
					hashPairMap.put(hashtag_pair, count);							
					text.set(hashtag_pair);
					writableCount.set(score);
					context.write(text,writableCount); 
				}
				
			}

		}
	}
}
