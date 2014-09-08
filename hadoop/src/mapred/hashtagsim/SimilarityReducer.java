package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.IntWritable;

/* SimilarityReducer -- final reducer for Similarity Score
 *						accumulate all the scores for each hash pair
 */

public class SimilarityReducer extends
		Reducer<Text, IntWritable, IntWritable, Text> {

	@Override
	protected void reduce(Text key, Iterable<IntWritable> value, Context context)
			throws IOException, InterruptedException {
		//if we do not use hash tag, their will be a bug
		Map<Text, Integer> counts = new HashMap<Text, Integer>();
		for (IntWritable scoreWritable : value) {
			Integer score = scoreWritable.get();
			Integer count = counts.get(key);
			if (count == null)
				count = 0;
			count = count + score;
			counts.put(key, count);
		}
		IntWritable result = new IntWritable();
		for (Map.Entry<Text, Integer> entry : counts.entrySet()) {
			Text text = entry.getKey();
			Integer out = entry.getValue();
			result.set(out);
			context.write(result, text);
		}

	}
}
