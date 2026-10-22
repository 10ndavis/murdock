import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'home_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
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
          scaffoldBackgroundColor: Colors.black,
          colorScheme: const ColorScheme.dark(primary: Color(0xFFFF2457)),
          fontFamily: 'monospace',
        ),
        home: const HomePage(),
      );
}
