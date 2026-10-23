import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'home_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Upside down on purpose: the speaker (normally the bottom edge) ends up on top, so a
  // comfortable viewing angle aims it away from you. See ui/grip.dart.
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitDown]);
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  runApp(const MurdockApp());
}

class MurdockApp extends StatelessWidget {
  const MurdockApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Murdock',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.dark,
          scaffoldBackgroundColor: const Color(0xFF05060A),
          colorScheme: const ColorScheme.dark(primary: Color(0xFFFF2457), surface: Color(0xFF05060A)),
          fontFamily: 'monospace',
        ),
        home: const HomePage(),
      );
}
