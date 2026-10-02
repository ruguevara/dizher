// Image to ZX Spec 2.3.0 (KodeMunkie) without its window, as its WorkProcessor converts one picture: scale, contrast,
// saturation, brightness at its defaults, then the chosen dither with its default colour and attribute modes.
//   java -Djava.awt.headless=true -cp imagetozxspec-2.3.0.jar Izx.java IN OUT ed|ordered INDEX
// ed: 0 Atkinson (its default), 2 Floyd-Steinberg; ordered: 3 Bayer 4x4, 4 Bayer 8x8 (OptionsObject's lists)
import java.io.File;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import uk.co.silentsoftware.config.OptionsObject;
import uk.co.silentsoftware.core.helpers.ColourHelper;
import uk.co.silentsoftware.core.helpers.ImageHelper;
import uk.co.silentsoftware.core.converters.image.processors.*;

public class Izx {
	public static void main(String[] a) throws Exception {
		OptionsObject oo = OptionsObject.getInstance();
		boolean ed = a[2].equals("ed");
		int i = Integer.parseInt(a[3]);
		oo.setSelectedDitherStrategy(ed ? oo.getErrorDithers()[i] : oo.getOrderedDithers()[i]);
		BufferedImage s = ImageHelper.quickScaleImage(ImageIO.read(new File(a[0])), 256, 192);
		s = ColourHelper.changeContrast(s, oo.getContrast());
		s = ColourHelper.changeSaturation(s, oo.getSaturation());
		s = ColourHelper.changeBrightness(s, oo.getBrightness());
		ImageConverter c = ed ? new ErrorDiffusionConverterImpl() : new OrderedDitherConverterImpl();
		ImageIO.write(c.convert(s)[0].getImage(), "png", new File(a[1]));
	}
}
