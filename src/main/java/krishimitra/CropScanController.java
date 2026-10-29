

package krishimitra;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Map;

@RestController
@RequestMapping("/api/crop-scans")
@CrossOrigin(origins = "http://localhost:5173")
public class CropScanController {

    private final CropDiseaseService cropDiseaseService;

    public CropScanController(CropDiseaseService cropDiseaseService) {
        this.cropDiseaseService = cropDiseaseService;
    }

    @PostMapping(
        value = "/upload",
        consumes = MediaType.MULTIPART_FORM_DATA_VALUE
    )
    public ResponseEntity<Map<String, Object>> uploadImage(
            @RequestParam("file") MultipartFile file) {

        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(
                Map.of("message", "Please select an image.")
            );
        }

        if (file.getContentType() == null ||
                !file.getContentType().startsWith("image/")) {
            return ResponseEntity.badRequest().body(
                Map.of("message", "Please upload a valid image file.")
            );
        }

        try {
            BufferedImage image = ImageIO.read(file.getInputStream());

            if (image == null) {
                return ResponseEntity.badRequest().body(
                    Map.of("message", "The uploaded file is not a readable image.")
                );
            }

            Map<String, Object> prediction =
                cropDiseaseService.predict(image);

            return ResponseEntity.ok(prediction);

        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(
                Map.of("message", "Unable to read the uploaded image.")
            );
        } catch (Exception e) {
            e.printStackTrace();

            return ResponseEntity.status(
                HttpStatus.INTERNAL_SERVER_ERROR
            ).body(
                Map.of("message", "Disease detection failed. Check backend logs.")
            );
        }
    }
}
